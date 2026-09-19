#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kQueryWords = 6;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::uint64_t exact_file_size(const fs::path &path) {
    std::error_code error;
    const auto size = fs::file_size(path, error);
    if(error) fail("cannot stat " + path.string() + ": " + error.message());
    return size;
}

template <typename T>
class ReadOnlyMap {
public:
    ReadOnlyMap(const fs::path &path, std::uint64_t count) : count_(count) {
        if(exact_file_size(path) != count * sizeof(T)) {
            fail("byte size mismatch for " + path.string());
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) fail("cannot open " + path.string() + ": " + std::strerror(errno));
        if(count != 0) {
            data_ = static_cast<const T *>(
                ::mmap(nullptr, static_cast<std::size_t>(count * sizeof(T)), PROT_READ,
                       MAP_PRIVATE, fd_, 0));
            if(data_ == MAP_FAILED) {
                data_ = nullptr;
                fail("cannot mmap " + path.string() + ": " + std::strerror(errno));
            }
        }
    }
    ReadOnlyMap(const ReadOnlyMap &) = delete;
    ReadOnlyMap &operator=(const ReadOnlyMap &) = delete;
    ~ReadOnlyMap() {
        if(data_ != nullptr) {
            ::munmap(const_cast<T *>(data_), static_cast<std::size_t>(count_ * sizeof(T)));
        }
        if(fd_ >= 0) ::close(fd_);
    }
    const T &operator[](std::uint64_t index) const {
        if(index >= count_) fail("mapped access is out of bounds");
        return data_[index];
    }

private:
    int fd_ = -1;
    const T *data_ = nullptr;
    std::uint64_t count_ = 0;
};

struct Group {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::array<std::uint64_t, kWords> union_words{};
};

struct Query {
    std::uint64_t id = 0;
    std::array<std::uint64_t, kWords> words{};
    std::uint16_t popcount = 0;
};

std::vector<std::array<std::uint64_t, kWords>> read_unions(
    const fs::path &path, const std::vector<std::uint64_t> &offsets) {
    std::ifstream input(path);
    if(!input) fail("cannot open TGM group statistics");
    std::string line;
    if(!std::getline(input, line) || line !=
       "group\trows\tunion_popcount\tword0\tword1\tword2\tword3") {
        fail("unexpected TGM group-statistics header");
    }
    std::vector<std::array<std::uint64_t, kWords>> unions;
    while(std::getline(input, line)) {
        std::istringstream stream(line);
        std::array<std::string, 7> fields{};
        for(auto &field : fields) {
            if(!std::getline(stream, field, '\t')) fail("short TGM group row");
        }
        if(std::stoull(fields[0]) != unions.size()) fail("non-consecutive TGM group ID");
        if(unions.size() + 1 >= offsets.size()) fail("too many TGM groups");
        if(std::stoull(fields[1]) != offsets[unions.size() + 1] - offsets[unions.size()]) {
            fail("TGM group size disagrees with offsets");
        }
        std::array<std::uint64_t, kWords> words{};
        unsigned popcount = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            words[word] = std::stoull(fields[3 + word], nullptr, 16);
            popcount += static_cast<unsigned>(std::popcount(words[word]));
        }
        if(popcount != std::stoul(fields[2])) fail("TGM union popcount mismatch");
        unions.push_back(words);
    }
    if(unions.size() + 1 != offsets.size()) fail("TGM group count disagrees with offsets");
    return unions;
}

std::vector<Query> read_queries(const fs::path &path, std::size_t limit) {
    const auto bytes = exact_file_size(path);
    if(bytes == 0 || bytes % (kQueryWords * sizeof(std::uint64_t)) != 0) {
        fail("invalid packed query file size");
    }
    const auto available = bytes / (kQueryWords * sizeof(std::uint64_t));
    const auto count = std::min<std::uint64_t>(available, limit);
    std::ifstream input(path, std::ios::binary);
    if(!input) fail("cannot open packed queries");
    std::vector<Query> queries(count);
    for(auto &query : queries) {
        std::array<std::uint64_t, kQueryWords> packed{};
        input.read(reinterpret_cast<char *>(packed.data()), sizeof(packed));
        if(input.gcount() != static_cast<std::streamsize>(sizeof(packed))) fail("short query read");
        query.id = packed[0];
        unsigned popcount = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            query.words[word] = packed[1 + word];
            popcount += static_cast<unsigned>(std::popcount(query.words[word]));
        }
        if(packed[5] == 0 || packed[5] != popcount) fail("query popcount mismatch");
        query.popcount = static_cast<std::uint16_t>(packed[5]);
    }
    return queries;
}

std::uint64_t hash_ids(const std::vector<std::uint64_t> &ids) {
    std::uint64_t hash = 1469598103934665603ULL;
    for(const auto id : ids) {
        const auto *bytes = reinterpret_cast<const unsigned char *>(&id);
        for(std::size_t index = 0; index < sizeof(id); ++index) {
            hash ^= bytes[index];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

struct Result {
    std::uint64_t candidate_groups = 0;
    std::uint64_t candidate_rows = 0;
    std::uint64_t hits = 0;
    std::uint64_t id_hash = 0;
    bool duplicate_id = false;
    double scan_materialize_seconds = 0;
    double sort_hash_seconds = 0;
};

class SearchIndex {
public:
    SearchIndex(std::uint64_t rows,
                const ReadOnlyMap<std::uint64_t> &source_fp,
                const ReadOnlyMap<std::uint64_t> &source_ids,
                const ReadOnlyMap<std::uint16_t> &source_popcounts,
                const ReadOnlyMap<std::uint32_t> &order,
                std::vector<std::uint64_t> offsets,
                std::vector<std::array<std::uint64_t, kWords>> unions)
        : rows_(rows), offsets_(std::move(offsets)), unions_(std::move(unions)),
          fingerprints_(rows * kWords), ids_(rows), popcounts_(rows) {
        std::vector<std::uint8_t> seen((rows + 7) / 8, 0);
        for(std::uint64_t position = 0; position < rows; ++position) {
            const auto row = order[position];
            if(row >= rows) fail("partition contains an out-of-range row");
            const auto byte = row >> 3U;
            const auto mask = static_cast<std::uint8_t>(1U << (row & 7U));
            if((seen[byte] & mask) != 0) fail("partition contains a duplicate row");
            seen[byte] |= mask;
            for(std::size_t word = 0; word < kWords; ++word) {
                fingerprints_[position * kWords + word] = source_fp[row * kWords + word];
            }
            ids_[position] = source_ids[row];
            popcounts_[position] = source_popcounts[row];
            unsigned observed = 0;
            for(std::size_t word = 0; word < kWords; ++word) {
                observed += static_cast<unsigned>(
                    std::popcount(fingerprints_[position * kWords + word]));
            }
            if(observed != popcounts_[position]) fail("candidate popcount mismatch");
        }
        groups_.reserve(unions_.size());
        for(std::size_t group = 0; group < unions_.size(); ++group) {
            groups_.push_back({offsets_[group], offsets_[group + 1], unions_[group]});
        }
    }

    Result search(const Query &query, unsigned numerator, unsigned denominator) const {
        const auto begin_time = std::chrono::steady_clock::now();
        Result result;
        std::vector<std::uint64_t> hit_ids;
        hit_ids.reserve(1024);
        for(const auto &group : groups_) {
            unsigned common_with_group = 0;
            for(std::size_t word = 0; word < kWords; ++word) {
                common_with_group += static_cast<unsigned>(
                    std::popcount(query.words[word] & group.union_words[word]));
            }
            if(static_cast<std::uint64_t>(denominator) * common_with_group <
               static_cast<std::uint64_t>(numerator) * query.popcount) {
                continue;
            }
            ++result.candidate_groups;
            result.candidate_rows += group.end - group.begin;
            for(auto position = group.begin; position < group.end; ++position) {
                unsigned intersection = 0;
                for(std::size_t word = 0; word < kWords; ++word) {
                    intersection += static_cast<unsigned>(std::popcount(
                        query.words[word] & fingerprints_[position * kWords + word]));
                }
                const unsigned union_count = query.popcount + popcounts_[position] - intersection;
                if(static_cast<std::uint64_t>(denominator) * intersection >=
                   static_cast<std::uint64_t>(numerator) * union_count) {
                    hit_ids.push_back(ids_[position]);
                }
            }
        }
        const auto materialized_time = std::chrono::steady_clock::now();
        result.hits = hit_ids.size();
        std::sort(hit_ids.begin(), hit_ids.end());
        result.duplicate_id = std::adjacent_find(hit_ids.begin(), hit_ids.end()) != hit_ids.end();
        result.id_hash = hash_ids(hit_ids);
        const auto finish_time = std::chrono::steady_clock::now();
        result.scan_materialize_seconds =
            std::chrono::duration<double>(materialized_time - begin_time).count();
        result.sort_hash_seconds =
            std::chrono::duration<double>(finish_time - materialized_time).count();
        return result;
    }

private:
    std::uint64_t rows_;
    std::vector<std::uint64_t> offsets_;
    std::vector<std::array<std::uint64_t, kWords>> unions_;
    std::vector<Group> groups_;
    std::vector<std::uint64_t> fingerprints_;
    std::vector<std::uint64_t> ids_;
    std::vector<std::uint16_t> popcounts_;
};

}  // namespace

int main(int argc, char **argv) {
    try {
        if constexpr(std::endian::native != std::endian::little) {
            fail("only little-endian hosts are supported");
        }
        if(argc != 15 || std::string(argv[9]) != "--rows" ||
           std::string(argv[11]) != "--query-limit" ||
           std::string(argv[13]) != "--warmup") {
            std::cerr << "usage: tgm_exact_search FP IDS POP ORDER OFFSETS GROUPS_TSV "
                         "QUERIES OUTPUT_CSV --rows N --query-limit N --warmup N\n";
            return 2;
        }
        const auto rows = std::stoull(argv[10]);
        const auto query_limit = std::stoull(argv[12]);
        const auto warmup = std::stoull(argv[14]);
        if(rows == 0 || rows > std::numeric_limits<std::uint32_t>::max()) fail("invalid rows");
        const fs::path output_path = argv[8];
        if(fs::exists(output_path)) fail("refusing to replace output CSV");
        ReadOnlyMap<std::uint64_t> source_fp(argv[1], rows * kWords);
        ReadOnlyMap<std::uint64_t> source_ids(argv[2], rows);
        ReadOnlyMap<std::uint16_t> source_popcounts(argv[3], rows);
        ReadOnlyMap<std::uint32_t> order(argv[4], rows);
        const auto offsets_bytes = exact_file_size(argv[5]);
        if(offsets_bytes < 16 || offsets_bytes % sizeof(std::uint64_t) != 0) {
            fail("invalid offsets file");
        }
        ReadOnlyMap<std::uint64_t> mapped_offsets(argv[5], offsets_bytes / 8);
        std::vector<std::uint64_t> offsets(offsets_bytes / 8);
        for(std::size_t index = 0; index < offsets.size(); ++index) offsets[index] = mapped_offsets[index];
        if(offsets.front() != 0 || offsets.back() != rows) fail("offsets do not span all rows");
        const auto unions = read_unions(argv[6], offsets);
        const auto queries = read_queries(argv[7], query_limit);
        const auto build_begin = std::chrono::steady_clock::now();
        SearchIndex index(rows, source_fp, source_ids, source_popcounts, order,
                          offsets, unions);
        const double reorder_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - build_begin).count();
        constexpr std::array<std::pair<unsigned, unsigned>, 2> thresholds{{{7, 10}, {4, 5}}};
        for(std::size_t iteration = 0; iteration < warmup; ++iteration) {
            const auto &query = queries[iteration % queries.size()];
            for(const auto [numerator, denominator] : thresholds) {
                const auto result = index.search(query, numerator, denominator);
                if(result.duplicate_id) fail("duplicate ID during warmup");
            }
        }
        std::ofstream output(output_path);
        if(!output) fail("cannot create output CSV");
        output << "query,query_id,threshold_num,threshold_den,candidate_groups,candidate_rows,"
                  "hits,id_hash,duplicate_id,scan_materialize_seconds,sort_hash_seconds,total_seconds\n";
        std::uint64_t duplicate_requests = 0;
        for(std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
            const auto &query = queries[query_index];
            for(const auto [numerator, denominator] : thresholds) {
                const auto result = index.search(query, numerator, denominator);
                duplicate_requests += result.duplicate_id;
                output << query_index << ',' << query.id << ',' << numerator << ',' << denominator
                       << ',' << result.candidate_groups << ',' << result.candidate_rows
                       << ',' << result.hits << ',' << result.id_hash << ','
                       << result.duplicate_id << ',' << std::setprecision(17)
                       << result.scan_materialize_seconds << ',' << result.sort_hash_seconds << ','
                       << result.scan_materialize_seconds + result.sort_hash_seconds << '\n';
            }
        }
        if(!output) fail("failed to write output CSV");
        std::cout << std::setprecision(17)
                  << "PASS rows=" << rows << " groups=" << unions.size()
                  << " queries=" << queries.size() << " requests=" << queries.size() * 2
                  << " duplicate_requests=" << duplicate_requests
                  << " reorder_seconds=" << reorder_seconds << '\n';
        return duplicate_requests == 0 ? 0 : 3;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}

