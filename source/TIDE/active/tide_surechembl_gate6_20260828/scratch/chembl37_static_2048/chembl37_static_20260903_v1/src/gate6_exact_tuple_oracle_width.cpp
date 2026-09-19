#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path &path) : path_(path.string()) {
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path_);
    struct stat state {};
    if (fstat(fd_, &state) != 0)
      throw std::runtime_error("fstat failed: " + path_);
    bytes_ = static_cast<std::size_t>(state.st_size);
    if (bytes_) {
      data_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
      if (data_ == MAP_FAILED)
        throw std::runtime_error("mmap failed: " + path_);
    }
  }
  ~MappedFile() {
    if (data_ != MAP_FAILED) munmap(data_, bytes_);
    if (fd_ >= 0) close(fd_);
  }
  template <class T>
  const T *as() const {
    if (bytes_ % sizeof(T))
      throw std::runtime_error("element-size mismatch: " + path_);
    return static_cast<const T *>(data_);
  }
  std::size_t bytes() const { return bytes_; }

 private:
  std::string path_;
  int fd_ = -1;
  std::size_t bytes_ = 0;
  void *data_ = MAP_FAILED;
};

struct Query {
  std::uint64_t id = 0;
  std::vector<std::uint64_t> fp;
  std::uint16_t popcount = 0;
};

struct Hit {
  std::uint64_t id = 0;
  std::uint16_t intersection = 0;
  std::uint16_t union_count = 0;
};

std::vector<Query> load_queries(const MappedFile &file, int words, int limit) {
  const int row_words = words + 2;
  if (file.bytes() % (row_words * sizeof(std::uint64_t)))
    throw std::runtime_error("invalid query file width");
  const auto *packed = file.as<std::uint64_t>();
  const std::size_t available =
      file.bytes() / (row_words * sizeof(std::uint64_t));
  const std::size_t count = std::min<std::size_t>(available, limit);
  std::vector<Query> queries(count);
  for (std::size_t row = 0; row < count; ++row) {
    auto &query = queries[row];
    query.id = packed[row * row_words];
    query.fp.resize(words);
    unsigned actual = 0;
    for (int word = 0; word < words; ++word) {
      query.fp[word] = packed[row * row_words + 1 + word];
      actual += __builtin_popcountll(query.fp[word]);
    }
    const auto stored = packed[row * row_words + row_words - 1];
    if (stored != actual) throw std::runtime_error("query popcount mismatch");
    query.popcount = static_cast<std::uint16_t>(stored);
  }
  return queries;
}

void sort_hits(std::vector<Hit> &hits) {
  std::sort(hits.begin(), hits.end(), [](const Hit &left, const Hit &right) {
    if (left.id != right.id) return left.id < right.id;
    if (left.intersection != right.intersection)
      return left.intersection < right.intersection;
    return left.union_count < right.union_count;
  });
}

std::uint64_t hash_hits(const std::vector<Hit> &hits) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto &hit : hits) {
    const std::uint64_t values[2] = {
        hit.id, static_cast<std::uint64_t>(hit.intersection) |
                    (static_cast<std::uint64_t>(hit.union_count) << 16)};
    const auto *bytes = reinterpret_cast<const unsigned char *>(values);
    for (std::size_t index = 0; index < sizeof(values); ++index) {
      hash ^= bytes[index];
      hash *= 1099511628211ull;
    }
  }
  return hash;
}

struct Record {
  int query = 0;
  int numerator = 0;
  int denominator = 0;
  std::uint64_t candidates = 0;
  std::uint64_t hits = 0;
  std::uint64_t result_hash = 0;
  bool duplicate_id = false;
  double seconds = 0;
};

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      std::cerr << "usage: " << argv[0]
                << " ROOT OUTPUT_CSV QUERY_LIMIT WORDS\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path output_path = argv[2];
    const int query_limit = std::stoi(argv[3]);
    const int words = std::stoi(argv[4]);
    if (words != 4 && words != 32)
      throw std::runtime_error("WORDS must be 4 or 32");

    const auto gate0 = root / "data/gate0_prepared";
    const auto stage_a = root / "data/stage_a";
    MappedFile fp_file(gate0 / ("union_fp_u64x" + std::to_string(words) + ".bin"));
    MappedFile id_file(gate0 / "union_id_i64.bin");
    MappedFile pc_file(gate0 / "union_popcnt_u16.bin");
    MappedFile query_file(
        stage_a / ("queries_u64x" + std::to_string(words + 2) + ".bin"));
    const std::uint64_t rows = id_file.bytes() / sizeof(std::uint64_t);
    if (fp_file.bytes() != rows * words * sizeof(std::uint64_t) ||
        pc_file.bytes() != rows * sizeof(std::uint16_t))
      throw std::runtime_error("union component-size mismatch");
    const auto *fp = fp_file.as<std::uint64_t>();
    const auto *ids = id_file.as<std::uint64_t>();
    const auto *pc = pc_file.as<std::uint16_t>();
    std::vector<std::uint64_t> bins(words * 64 + 2, 0);
    std::uint16_t previous = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
      if (pc[row] > words * 64 || (row && pc[row] < previous))
        throw std::runtime_error("union population-count order mismatch");
      bins[pc[row] + 1]++;
      previous = pc[row];
    }
    for (std::size_t index = 1; index < bins.size(); ++index)
      bins[index] += bins[index - 1];
    const auto queries = load_queries(query_file, words, query_limit);
    const std::vector<std::pair<int, int>> thresholds{{7, 10}, {4, 5}};
    std::vector<Record> records(queries.size() * thresholds.size());

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t task = 0;
         task < static_cast<std::int64_t>(records.size()); ++task) {
      const auto begin = std::chrono::steady_clock::now();
      const int query_index = static_cast<int>(task / thresholds.size());
      const int threshold_index = static_cast<int>(task % thresholds.size());
      const auto [num, den] = thresholds[threshold_index];
      const auto &query = queries[query_index];
      const int lower =
          (num * static_cast<int>(query.popcount) + den - 1) / den;
      const int upper = den * static_cast<int>(query.popcount) / num;
      const std::uint64_t first = bins[std::max(0, lower)];
      const std::uint64_t last = bins[std::min(words * 64, upper) + 1];
      std::vector<Hit> hits;
      for (std::uint64_t row = first; row < last; ++row) {
        unsigned intersection = 0;
        for (int word = 0; word < words; ++word)
          intersection += __builtin_popcountll(
              query.fp[word] & fp[row * words + word]);
        unsigned union_count = static_cast<unsigned>(query.popcount) + pc[row] -
                               intersection;
        if (union_count == 0) union_count = 1;
        if (intersection * static_cast<unsigned>(den) <
            union_count * static_cast<unsigned>(num))
          continue;
        hits.push_back(Hit{ids[row], static_cast<std::uint16_t>(intersection),
                           static_cast<std::uint16_t>(union_count)});
      }
      sort_hits(hits);
      const bool duplicate =
          std::adjacent_find(hits.begin(), hits.end(),
                             [](const Hit &left, const Hit &right) {
                               return left.id == right.id;
                             }) != hits.end();
      records[task] = Record{
          query_index,
          num,
          den,
          last - first,
          hits.size(),
          hash_hits(hits),
          duplicate,
          std::chrono::duration<double>(std::chrono::steady_clock::now() - begin)
              .count()};
    }

    if (!output_path.parent_path().empty())
      std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    output << "query,threshold_num,threshold_den,candidate_rows,hits,result_hash,"
              "duplicate_id,diagnostic_cpu_seconds\n";
    std::uint64_t duplicates = 0;
    for (const auto &record : records) {
      duplicates += record.duplicate_id;
      output << record.query << ',' << record.numerator << ','
             << record.denominator << ',' << record.candidates << ','
             << record.hits << ',' << record.result_hash << ','
             << record.duplicate_id << ',' << std::setprecision(17)
             << record.seconds << '\n';
    }
    std::ofstream summary(output_path.string() + ".summary.json");
    summary << "{\n"
            << "  \"experiment_id\": "
               "\"tide_20260903_exact_tuple_oracle_width_v1\",\n"
            << "  \"rows\": " << rows << ",\n"
            << "  \"word_count\": " << words << ",\n"
            << "  \"queries\": " << queries.size() << ",\n"
            << "  \"requests\": " << records.size() << ",\n"
            << "  \"duplicate_id_requests\": " << duplicates << ",\n"
            << "  \"exact_rational_oracle_pass\": "
            << (duplicates == 0 ? "true" : "false") << "\n}\n";
    return duplicates == 0 ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
