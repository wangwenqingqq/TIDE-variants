#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <QCoreApplication>
#include <QString>

#include "fingerprintdb_cuda.h"

namespace {

constexpr std::size_t kFingerprintWords = 4;
constexpr std::size_t kQueryWords = 6;
constexpr std::size_t kIdHexBytes = 17;

[[noreturn]] void fail(const std::string &message) {
    throw std::runtime_error(message);
}

std::uint64_t exact_file_size(const std::string &path) {
    struct stat status {};
    if(::stat(path.c_str(), &status) != 0) {
        fail("cannot stat " + path + ": " + std::strerror(errno));
    }
    if(status.st_size < 0) fail("negative file size for " + path);
    return static_cast<std::uint64_t>(status.st_size);
}

template <typename T>
class ReadOnlyMap {
public:
    ReadOnlyMap(const std::string &path, std::uint64_t count) : count_(count) {
        if(exact_file_size(path) != count * sizeof(T)) {
            fail("byte size mismatch for " + path);
        }
        fd_ = ::open(path.c_str(), O_RDONLY);
        if(fd_ < 0) fail("cannot open " + path + ": " + std::strerror(errno));
        if(count_ != 0) {
            void *mapped = ::mmap(nullptr, static_cast<std::size_t>(count_ * sizeof(T)),
                                  PROT_READ, MAP_PRIVATE, fd_, 0);
            if(mapped == MAP_FAILED) {
                fail("cannot mmap " + path + ": " + std::strerror(errno));
            }
            data_ = static_cast<const T *>(mapped);
        }
    }

    ReadOnlyMap(const ReadOnlyMap &) = delete;
    ReadOnlyMap &operator=(const ReadOnlyMap &) = delete;

    ~ReadOnlyMap() {
        if(data_ != nullptr) {
            ::munmap(const_cast<T *>(data_),
                     static_cast<std::size_t>(count_ * sizeof(T)));
        }
        if(fd_ >= 0) ::close(fd_);
    }

    const T *data() const { return data_; }
    const T &operator[](std::uint64_t index) const {
        if(index >= count_) fail("mapped access is out of bounds");
        return data_[index];
    }

private:
    int fd_ = -1;
    const T *data_ = nullptr;
    std::uint64_t count_ = 0;
};

struct Query {
    std::uint64_t id = 0;
    gpusim::Fingerprint words;
};

std::vector<Query> read_queries(const std::string &path, std::size_t limit) {
    const auto bytes = exact_file_size(path);
    if(bytes == 0 || bytes % (kQueryWords * sizeof(std::uint64_t)) != 0) {
        fail("invalid packed query file size");
    }
    const auto available = bytes / (kQueryWords * sizeof(std::uint64_t));
    const auto count = std::min<std::uint64_t>(available, limit);
    ReadOnlyMap<std::uint64_t> packed(path, available * kQueryWords);
    std::vector<Query> queries;
    queries.reserve(static_cast<std::size_t>(count));
    for(std::uint64_t query_index = 0; query_index < count; ++query_index) {
        const auto offset = query_index * kQueryWords;
        Query query;
        query.id = packed[offset];
        query.words.resize(8);
        std::memcpy(query.words.data(), packed.data() + offset + 1,
                    kFingerprintWords * sizeof(std::uint64_t));
        unsigned popcount = 0;
        for(std::size_t word = 0; word < kFingerprintWords; ++word) {
            popcount += static_cast<unsigned>(
                __builtin_popcountll(packed[offset + 1 + word]));
        }
        if(packed[offset + 5] == 0 || packed[offset + 5] != popcount) {
            fail("query popcount mismatch");
        }
        queries.push_back(std::move(query));
    }
    return queries;
}

void encode_hex_id(std::uint64_t value, char *destination) {
    static const char digits[] = "0123456789abcdef";
    for(std::size_t index = 0; index < 16; ++index) {
        const unsigned shift = static_cast<unsigned>((15 - index) * 4);
        destination[index] = digits[(value >> shift) & 0xfU];
    }
    destination[16] = '\0';
}

std::uint64_t decode_hex_id(const char *source) {
    std::uint64_t value = 0;
    for(std::size_t index = 0; index < 16; ++index) {
        const unsigned char byte = static_cast<unsigned char>(source[index]);
        unsigned digit = 0;
        if(byte >= '0' && byte <= '9') {
            digit = byte - '0';
        } else if(byte >= 'a' && byte <= 'f') {
            digit = byte - 'a' + 10;
        } else {
            fail("invalid stable-ID encoding returned by GPUSimilarity");
        }
        value = (value << 4U) | digit;
    }
    if(source[16] != '\0') fail("unterminated stable-ID encoding");
    return value;
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

std::vector<std::pair<unsigned, unsigned>> parse_threshold_order(
    const std::string &value) {
    std::vector<std::pair<unsigned, unsigned>> result;
    std::size_t begin = 0;
    while(begin < value.size()) {
        const auto comma = value.find(',', begin);
        const auto token = value.substr(
            begin, comma == std::string::npos ? std::string::npos : comma - begin);
        const auto slash = token.find('/');
        if(slash == std::string::npos || token.find('/', slash + 1) != std::string::npos) {
            fail("invalid threshold-order token: " + token);
        }
        const auto numerator = static_cast<unsigned>(std::stoul(token.substr(0, slash)));
        const auto denominator = static_cast<unsigned>(std::stoul(token.substr(slash + 1)));
        if(numerator == 0 || denominator == 0 || numerator > denominator) {
            fail("invalid rational threshold: " + token);
        }
        result.emplace_back(numerator, denominator);
        if(comma == std::string::npos) break;
        begin = comma + 1;
    }
    if(result.size() != 2 || result[0] == result[1]) {
        fail("threshold order must contain two distinct rational thresholds");
    }
    return result;
}

struct SearchResult {
    unsigned long approximate_count = 0;
    std::size_t returned_count = 0;
    bool duplicate_id = false;
    std::uint64_t id_hash = 0;
    float minimum_score = 0.0F;
    double search_seconds = 0.0;
    double validation_seconds = 0.0;
};

SearchResult run_search(const gpusim::FingerprintDB &database,
                        const gpusim::Fingerprint &query, unsigned cap,
                        unsigned numerator, unsigned denominator) {
    std::vector<char *> result_smiles;
    std::vector<char *> result_ids;
    std::vector<float> result_scores;
    unsigned long approximate_count = 0;
    const float cutoff = static_cast<float>(numerator) /
                         static_cast<float>(denominator);
    const auto begin = std::chrono::steady_clock::now();
    database.search(query, QString(), cap, cutoff, result_smiles, result_ids,
                    result_scores, approximate_count);
    const auto searched = std::chrono::steady_clock::now();

    if(result_smiles.size() != result_ids.size() ||
       result_ids.size() != result_scores.size()) {
        fail("GPUSimilarity returned inconsistent result-vector sizes");
    }
    if(approximate_count >= cap) {
        fail("result count reaches the cap; complete output is not proven");
    }
    if(approximate_count != result_ids.size()) {
        fail("approximate count differs from complete returned count");
    }
    std::vector<std::uint64_t> numeric_ids;
    numeric_ids.reserve(result_ids.size());
    for(std::size_t index = 0; index < result_ids.size(); ++index) {
        if(result_scores[index] < cutoff) {
            fail("GPUSimilarity returned a score below its cutoff");
        }
        numeric_ids.push_back(decode_hex_id(result_ids[index]));
    }
    std::sort(numeric_ids.begin(), numeric_ids.end());
    const bool duplicate =
        std::adjacent_find(numeric_ids.begin(), numeric_ids.end()) != numeric_ids.end();
    const auto id_hash = hash_ids(numeric_ids);
    const auto finished = std::chrono::steady_clock::now();

    SearchResult result;
    result.approximate_count = approximate_count;
    result.returned_count = result_ids.size();
    result.duplicate_id = duplicate;
    result.id_hash = id_hash;
    result.minimum_score = result_scores.empty()
                               ? std::numeric_limits<float>::quiet_NaN()
                               : *std::min_element(result_scores.begin(), result_scores.end());
    result.search_seconds =
        std::chrono::duration<double>(searched - begin).count();
    result.validation_seconds =
        std::chrono::duration<double>(finished - searched).count();
    return result;
}

}  // namespace

int main(int argc, char **argv) {
    try {
        QCoreApplication application(argc, argv);
        const std::uint16_t endian_probe = 1;
        if(*reinterpret_cast<const unsigned char *>(&endian_probe) != 1) {
            fail("only little-endian hosts are supported");
        }
        if(argc != 15 || std::string(argv[5]) != "--rows" ||
           std::string(argv[7]) != "--query-limit" ||
           std::string(argv[9]) != "--warmup" ||
           std::string(argv[11]) != "--cap" ||
           std::string(argv[13]) != "--threshold-order") {
            std::cerr << "usage: gpusim_direct_exact FP_U64X4 IDS_I64 QUERIES_U64X6 "
                         "OUTPUT_CSV --rows N --query-limit N --warmup N --cap N "
                         "--threshold-order 7/10,4/5\n";
            return 2;
        }

        const std::string fingerprint_path = argv[1];
        const std::string id_path = argv[2];
        const std::string query_path = argv[3];
        const std::string output_path = argv[4];
        const auto rows_u64 = std::stoull(argv[6]);
        const auto query_limit = std::stoull(argv[8]);
        const auto warmup = std::stoull(argv[10]);
        const auto cap_u64 = std::stoull(argv[12]);
        const auto thresholds = parse_threshold_order(argv[14]);
        if(rows_u64 == 0 || rows_u64 > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
            fail("row count exceeds the released signed-int API");
        }
        if(query_limit == 0 || query_limit > std::numeric_limits<std::size_t>::max()) {
            fail("invalid query limit");
        }
        if(cap_u64 == 0 || cap_u64 > std::numeric_limits<unsigned>::max()) {
            fail("invalid result cap");
        }
        if(::access(output_path.c_str(), F_OK) == 0) {
            fail("refusing to replace output CSV");
        }
        const auto rows = static_cast<std::size_t>(rows_u64);
        const auto cap = static_cast<unsigned>(cap_u64);
        ReadOnlyMap<std::uint64_t> mapped_fingerprints(
            fingerprint_path, rows_u64 * kFingerprintWords);
        ReadOnlyMap<std::uint64_t> mapped_ids(id_path, rows_u64);
        const auto queries = read_queries(query_path,
                                          static_cast<std::size_t>(query_limit));

        std::vector<std::vector<char>> fingerprint_blocks(1);
        fingerprint_blocks[0].resize(rows * kFingerprintWords * sizeof(std::uint64_t));
        std::memcpy(fingerprint_blocks[0].data(), mapped_fingerprints.data(),
                    fingerprint_blocks[0].size());

        std::vector<char> id_arena(rows * kIdHexBytes);
        std::vector<char *> smiles;
        std::vector<char *> ids;
        smiles.reserve(rows);
        ids.reserve(rows);
        for(std::size_t row = 0; row < rows; ++row) {
            char *encoded = id_arena.data() + row * kIdHexBytes;
            encode_hex_id(mapped_ids[row], encoded);
            smiles.push_back(encoded);
            ids.push_back(encoded);
        }

        const auto construct_begin = std::chrono::steady_clock::now();
        gpusim::FingerprintDB database(256, static_cast<int>(rows_u64), QString(),
                                      fingerprint_blocks, smiles, ids);
        const auto constructed = std::chrono::steady_clock::now();
        std::vector<std::vector<char>>().swap(fingerprint_blocks);
        const auto copy_begin = std::chrono::steady_clock::now();
        database.copyToGPU(1);
        const auto copied = std::chrono::steady_clock::now();

        std::ofstream output(output_path);
        if(!output) fail("cannot create output CSV");
        output << "query,query_id,threshold_num,threshold_den,threshold_order_index,"
                  "approximate_count,returned_count,duplicate_id,id_hash,min_score,"
                  "search_seconds,validation_seconds\n";
        output << std::setprecision(17);

        for(std::size_t threshold_index = 0; threshold_index < thresholds.size();
            ++threshold_index) {
            const auto numerator = thresholds[threshold_index].first;
            const auto denominator = thresholds[threshold_index].second;
            for(std::size_t repetition = 0; repetition < warmup; ++repetition) {
                (void)run_search(database, queries.front().words, cap, numerator,
                                 denominator);
            }
            for(std::size_t query_index = 0; query_index < queries.size(); ++query_index) {
                const auto result = run_search(database, queries[query_index].words, cap,
                                               numerator, denominator);
                output << query_index << ',' << queries[query_index].id << ',' << numerator
                       << ',' << denominator << ',' << threshold_index << ','
                       << result.approximate_count << ',' << result.returned_count << ','
                       << (result.duplicate_id ? 1 : 0) << ',' << result.id_hash << ','
                       << result.minimum_score << ',' << result.search_seconds << ','
                       << result.validation_seconds << '\n';
            }
            output.flush();
            if(!output) fail("failed while writing output CSV");
        }

        struct rusage usage {};
        if(::getrusage(RUSAGE_SELF, &usage) != 0) {
            fail("getrusage failed");
        }
        std::cerr << std::setprecision(17)
                  << "construct_seconds="
                  << std::chrono::duration<double>(constructed - construct_begin).count()
                  << "\ncopy_to_gpu_seconds="
                  << std::chrono::duration<double>(copied - copy_begin).count()
                  << "\npeak_rss_kib=" << usage.ru_maxrss << '\n';
        return 0;
    } catch(const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}

