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
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace fs = std::filesystem;

namespace {

constexpr std::size_t kWords = 4;
constexpr std::size_t kPackedWords = 6;

[[noreturn]] void fail(const std::string &message) { throw std::runtime_error(message); }

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
        if(exact_file_size(path) != count * sizeof(T)) fail("byte size mismatch for " + path.string());
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) fail("cannot open " + path.string() + ": " + std::strerror(errno));
        if(count != 0) {
            data_ = static_cast<const T *>(::mmap(nullptr,
                static_cast<std::size_t>(count * sizeof(T)), PROT_READ, MAP_PRIVATE, fd_, 0));
            if(data_ == MAP_FAILED) {
                data_ = nullptr;
                fail("cannot mmap " + path.string() + ": " + std::strerror(errno));
            }
        }
    }
    ReadOnlyMap(const ReadOnlyMap &) = delete;
    ReadOnlyMap &operator=(const ReadOnlyMap &) = delete;
    ~ReadOnlyMap() {
        if(data_ != nullptr) ::munmap(const_cast<T *>(data_),
                                      static_cast<std::size_t>(count_ * sizeof(T)));
        if(fd_ >= 0) ::close(fd_);
    }
    const T &operator[](std::uint64_t index) const {
        if(index >= count_) fail("mapped access out of bounds");
        return data_[index];
    }

private:
    int fd_ = -1;
    const T *data_ = nullptr;
    std::uint64_t count_ = 0;
};

struct Row {
    std::array<std::uint64_t, kWords> words{};
    std::uint64_t id = 0;
    std::uint16_t popcount = 0;
};

struct Query : Row {};

struct Group {
    std::uint64_t begin = 0;
    std::uint64_t end = 0;
    std::array<std::uint64_t, kWords> base_union{};
    std::array<std::uint64_t, kWords> current_union{};
    std::vector<Row> inserted;
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
        if(std::stoull(fields[0]) != unions.size()) fail("non-consecutive group ID");
        if(unions.size() + 1 >= offsets.size()) fail("too many TGM groups");
        if(std::stoull(fields[1]) != offsets[unions.size() + 1] - offsets[unions.size()]) {
            fail("TGM group size disagrees with offsets");
        }
        std::array<std::uint64_t, kWords> words{};
        unsigned observed = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            words[word] = std::stoull(fields[3 + word], nullptr, 16);
            observed += static_cast<unsigned>(std::popcount(words[word]));
        }
        if(observed != std::stoul(fields[2])) fail("TGM union popcount mismatch");
        unions.push_back(words);
    }
    if(unions.size() + 1 != offsets.size()) fail("TGM group count mismatch");
    return unions;
}

std::vector<Row> read_packed_rows(const fs::path &path) {
    const auto bytes = exact_file_size(path);
    if(bytes == 0 || bytes % (kPackedWords * sizeof(std::uint64_t)) != 0) {
        fail("invalid packed-row file size");
    }
    std::ifstream input(path, std::ios::binary);
    if(!input) fail("cannot open packed rows");
    std::vector<Row> rows(bytes / (kPackedWords * sizeof(std::uint64_t)));
    for(auto &row : rows) {
        std::array<std::uint64_t, kPackedWords> packed{};
        input.read(reinterpret_cast<char *>(packed.data()), sizeof(packed));
        if(input.gcount() != static_cast<std::streamsize>(sizeof(packed))) fail("short packed-row read");
        row.id = packed[0];
        unsigned popcount = 0;
        for(std::size_t word = 0; word < kWords; ++word) {
            row.words[word] = packed[1 + word];
            popcount += static_cast<unsigned>(std::popcount(row.words[word]));
        }
        if(packed[5] != popcount || popcount == 0) fail("packed-row popcount mismatch");
        row.popcount = static_cast<std::uint16_t>(popcount);
    }
    return rows;
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

struct SearchResult {
    std::uint64_t candidate_groups = 0;
    std::uint64_t candidate_rows = 0;
    std::uint64_t hits = 0;
    std::uint64_t visible_query_id_count = 0;
    std::uint64_t id_hash = 0;
    bool duplicate_id = false;
    double materialize_seconds = 0;
    double sort_hash_seconds = 0;
};

struct UpdateStats {
    std::uint64_t group_checks = 0;
    std::uint64_t groups_receiving_rows = 0;
};

class DynamicIndex {
public:
    DynamicIndex(std::uint64_t rows,
                 const ReadOnlyMap<std::uint64_t> &source_fp,
                 const ReadOnlyMap<std::uint64_t> &source_ids,
                 const ReadOnlyMap<std::uint16_t> &source_popcounts,
                 const ReadOnlyMap<std::uint32_t> &order,
                 const std::vector<std::uint64_t> &offsets,
                 const std::vector<std::array<std::uint64_t, kWords>> &unions,
                 std::uint64_t visible_query_id)
        : fingerprints_(rows * kWords), ids_(rows), popcounts_(rows) {
        std::vector<std::uint8_t> seen((rows + 7) / 8, 0);
        std::uint64_t base_query_id_count = 0;
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
            base_query_id_count += ids_[position] == visible_query_id;
            unsigned observed = 0;
            for(std::size_t word = 0; word < kWords; ++word) {
                observed += static_cast<unsigned>(
                    std::popcount(fingerprints_[position * kWords + word]));
            }
            if(observed != popcounts_[position]) fail("base popcount mismatch");
        }
        if(base_query_id_count != 0) fail("future-arrival query ID is already visible in the base");
        groups_.reserve(unions.size());
        for(std::size_t group = 0; group < unions.size(); ++group) {
            groups_.push_back({offsets[group], offsets[group + 1], unions[group], unions[group], {}});
        }
    }

    void reset() {
        for(auto &group : groups_) {
            group.current_union = group.base_union;
            group.inserted.clear();
        }
    }

    UpdateStats insert_all(const std::vector<Row> &delta) {
        UpdateStats stats;
        std::vector<std::uint8_t> touched(groups_.size(), 0);
        for(const auto &row : delta) {
            unsigned best_common = 0;
            std::size_t best_group = 0;
            for(std::size_t group_index = 0; group_index < groups_.size(); ++group_index) {
                ++stats.group_checks;
                unsigned common = 0;
                for(std::size_t word = 0; word < kWords; ++word) {
                    common += static_cast<unsigned>(
                        std::popcount(row.words[word] & groups_[group_index].current_union[word]));
                }
                if(common > best_common) {
                    best_common = common;
                    best_group = group_index;
                }
                if(common == row.popcount) break;
            }
            auto &group = groups_[best_group];
            group.inserted.push_back(row);
            for(std::size_t word = 0; word < kWords; ++word) {
                group.current_union[word] |= row.words[word];
            }
            if(touched[best_group] == 0) {
                touched[best_group] = 1;
                ++stats.groups_receiving_rows;
            }
        }
        return stats;
    }

    SearchResult search(const Query &query, unsigned numerator, unsigned denominator) const {
        const auto begin = std::chrono::steady_clock::now();
        SearchResult result;
        std::vector<std::uint64_t> hits;
        hits.reserve(1024);
        for(const auto &group : groups_) {
            unsigned group_common = 0;
            for(std::size_t word = 0; word < kWords; ++word) {
                group_common += static_cast<unsigned>(
                    std::popcount(query.words[word] & group.current_union[word]));
            }
            if(static_cast<std::uint64_t>(denominator) * group_common <
               static_cast<std::uint64_t>(numerator) * query.popcount) continue;
            ++result.candidate_groups;
            result.candidate_rows += group.end - group.begin + group.inserted.size();
            for(auto position = group.begin; position < group.end; ++position) {
                unsigned intersection = 0;
                for(std::size_t word = 0; word < kWords; ++word) {
                    intersection += static_cast<unsigned>(std::popcount(
                        query.words[word] & fingerprints_[position * kWords + word]));
                }
                const unsigned union_count = query.popcount + popcounts_[position] - intersection;
                if(static_cast<std::uint64_t>(denominator) * intersection >=
                   static_cast<std::uint64_t>(numerator) * union_count) {
                    hits.push_back(ids_[position]);
                    result.visible_query_id_count += ids_[position] == query.id;
                }
            }
            for(const auto &row : group.inserted) {
                unsigned intersection = 0;
                for(std::size_t word = 0; word < kWords; ++word) {
                    intersection += static_cast<unsigned>(
                        std::popcount(query.words[word] & row.words[word]));
                }
                const unsigned union_count = query.popcount + row.popcount - intersection;
                if(static_cast<std::uint64_t>(denominator) * intersection >=
                   static_cast<std::uint64_t>(numerator) * union_count) {
                    hits.push_back(row.id);
                    result.visible_query_id_count += row.id == query.id;
                }
            }
        }
        const auto materialized = std::chrono::steady_clock::now();
        result.hits = hits.size();
        std::sort(hits.begin(), hits.end());
        result.duplicate_id = std::adjacent_find(hits.begin(), hits.end()) != hits.end();
        result.id_hash = hash_ids(hits);
        const auto finished = std::chrono::steady_clock::now();
        result.materialize_seconds =
            std::chrono::duration<double>(materialized - begin).count();
        result.sort_hash_seconds =
            std::chrono::duration<double>(finished - materialized).count();
        return result;
    }

private:
    std::vector<std::uint64_t> fingerprints_;
    std::vector<std::uint64_t> ids_;
    std::vector<std::uint16_t> popcounts_;
    std::vector<Group> groups_;
};

}  // namespace

int main(int argc, char **argv) {
    try {
        if constexpr(std::endian::native != std::endian::little) fail("only little-endian hosts are supported");
        if(argc != 20 || std::string(argv[10]) != "--base-rows" ||
           std::string(argv[12]) != "--query-index" ||
           std::string(argv[14]) != "--repetitions" ||
           std::string(argv[16]) != "--warmup" ||
           std::string(argv[18]) != "--threshold") {
            std::cerr << "usage: tgm_update_benchmark BASE_FP BASE_IDS BASE_POP ORDER OFFSETS "
                         "GROUPS_TSV DELTA_U64X6 QUERIES_U64X6 OUTPUT_CSV --base-rows N "
                         "--query-index N --repetitions N --warmup N --threshold NUM/DEN\n";
            return 2;
        }
        const auto base_rows = std::stoull(argv[11]);
        const auto query_index = std::stoull(argv[13]);
        const auto repetitions = std::stoull(argv[15]);
        const auto warmup = std::stoull(argv[17]);
        unsigned numerator = 0, denominator = 0;
        {
            const std::string threshold = argv[19];
            const auto slash = threshold.find('/');
            if(slash == std::string::npos) fail("threshold must be NUM/DEN");
            numerator = static_cast<unsigned>(std::stoul(threshold.substr(0, slash)));
            denominator = static_cast<unsigned>(std::stoul(threshold.substr(slash + 1)));
            if(numerator == 0 || denominator == 0 || numerator > denominator) fail("invalid threshold");
        }
        if(base_rows == 0 || repetitions == 0) fail("base rows and repetitions must be positive");
        const fs::path output_path = argv[9];
        if(fs::exists(output_path)) fail("refusing to replace output CSV");
        const auto delta = read_packed_rows(argv[7]);
        const auto packed_queries = read_packed_rows(argv[8]);
        if(query_index >= packed_queries.size()) fail("query index out of range");
        Query query;
        static_cast<Row &>(query) = packed_queries[query_index];
        std::uint64_t delta_query_id_count = 0;
        for(const auto &row : delta) delta_query_id_count += row.id == query.id;
        if(delta_query_id_count != 1) fail("future-arrival query ID must occur exactly once in delta");

        ReadOnlyMap<std::uint64_t> source_fp(argv[1], base_rows * kWords);
        ReadOnlyMap<std::uint64_t> source_ids(argv[2], base_rows);
        ReadOnlyMap<std::uint16_t> source_popcounts(argv[3], base_rows);
        ReadOnlyMap<std::uint32_t> order(argv[4], base_rows);
        const auto offsets_bytes = exact_file_size(argv[5]);
        if(offsets_bytes < 16 || offsets_bytes % 8 != 0) fail("invalid offsets file");
        ReadOnlyMap<std::uint64_t> mapped_offsets(argv[5], offsets_bytes / 8);
        std::vector<std::uint64_t> offsets(offsets_bytes / 8);
        for(std::size_t i = 0; i < offsets.size(); ++i) offsets[i] = mapped_offsets[i];
        if(offsets.front() != 0 || offsets.back() != base_rows) fail("offsets do not span base");
        const auto unions = read_unions(argv[6], offsets);
        DynamicIndex index(base_rows, source_fp, source_ids, source_popcounts,
                           order, offsets, unions, query.id);

        auto execute = [&]() {
            index.reset();
            const auto begin = std::chrono::steady_clock::now();
            const auto update_begin = begin;
            const auto update_stats = index.insert_all(delta);
            const auto update_end = std::chrono::steady_clock::now();
            const auto result = index.search(query, numerator, denominator);
            return std::tuple<UpdateStats, SearchResult, double, double>{
                update_stats, result,
                std::chrono::duration<double>(update_end - update_begin).count(),
                std::chrono::duration<double>(update_end - begin).count() +
                    result.materialize_seconds};
        };
        for(std::uint64_t i = 0; i < warmup; ++i) {
            const auto [stats, result, update_seconds, total_seconds] = execute();
            if(result.duplicate_id || result.visible_query_id_count != 1 ||
               result.hits == 0 || update_seconds <= 0 || total_seconds <= 0) {
                fail("warmup update/query gate failed");
            }
        }
        std::ofstream output(output_path);
        if(!output) fail("cannot create output CSV");
        output << "repetition,threshold_num,threshold_den,delta_rows,group_checks,"
                  "groups_receiving_rows,candidate_groups,candidate_rows,hits,"
                  "visible_query_id_count,id_hash,duplicate_id,update_seconds,"
                  "query_materialize_seconds,sort_hash_seconds,"
                  "update_to_visible_seconds\n";
        for(std::uint64_t repetition = 0; repetition < repetitions; ++repetition) {
            const auto [stats, result, update_seconds, total_seconds] = execute();
            output << repetition << ',' << numerator << ',' << denominator << ',' << delta.size()
                   << ',' << stats.group_checks << ',' << stats.groups_receiving_rows
                   << ',' << result.candidate_groups << ',' << result.candidate_rows
                   << ',' << result.hits << ',' << result.visible_query_id_count
                   << ',' << result.id_hash << ',' << result.duplicate_id
                   << ',' << std::setprecision(17) << update_seconds
                   << ',' << result.materialize_seconds << ',' << result.sort_hash_seconds
                   << ',' << total_seconds << '\n';
            if(result.duplicate_id) fail("duplicate output ID");
            if(result.visible_query_id_count != 1) {
                fail("future-arrival query ID is not visible exactly once");
            }
        }
        if(!output) fail("failed to write output CSV");
        std::cout << "PASS base_rows=" << base_rows << " delta_rows=" << delta.size()
                  << " groups=" << unions.size() << " query_index=" << query_index
                  << " repetitions=" << repetitions << " threshold=" << numerator << '/'
                  << denominator << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
