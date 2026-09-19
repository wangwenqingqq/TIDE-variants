#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

constexpr int WORDS = 4;
constexpr int ROW_WORDS = WORDS + 2;

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
  std::uint64_t id;
  std::array<std::uint64_t, WORDS> fp;
  std::uint16_t popcount;
};

std::vector<Query> load_queries(const MappedFile &file, int limit) {
  if (file.bytes() % (ROW_WORDS * sizeof(std::uint64_t)))
    throw std::runtime_error("invalid query file width");
  const auto *packed = file.as<std::uint64_t>();
  const std::size_t available = file.bytes() / (ROW_WORDS * sizeof(std::uint64_t));
  const std::size_t count = std::min<std::size_t>(available, limit);
  std::vector<Query> queries(count);
  for (std::size_t row = 0; row < count; ++row) {
    auto &query = queries[row];
    query.id = packed[row * ROW_WORDS];
    unsigned actual = 0;
    for (int word = 0; word < WORDS; ++word) {
      query.fp[word] = packed[row * ROW_WORDS + 1 + word];
      actual += __builtin_popcountll(query.fp[word]);
    }
    const auto stored = packed[row * ROW_WORDS + ROW_WORDS - 1];
    if (stored != actual) throw std::runtime_error("query popcount mismatch");
    query.popcount = static_cast<std::uint16_t>(stored);
  }
  return queries;
}

std::uint64_t hash_ids(const std::vector<std::uint64_t> &ids) {
  std::uint64_t hash = 1469598103934665603ull;
  for (const auto id : ids) {
    const auto *bytes = reinterpret_cast<const unsigned char *>(&id);
    for (std::size_t index = 0; index < sizeof(id); ++index) {
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
  std::uint64_t id_hash = 0;
  bool duplicate_id = false;
  double seconds = 0;
};

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: " << argv[0] << " TRANSITION_ROOT OUTPUT_CSV QUERY_LIMIT\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path output_path = argv[2];
    const int query_limit = std::stoi(argv[3]);
    MappedFile fp_file(root / "data/gate0_prepared/union_fp_u64x4.bin");
    MappedFile id_file(root / "data/gate0_prepared/union_id_i64.bin");
    MappedFile pc_file(root / "data/gate0_prepared/union_popcnt_u16.bin");
    MappedFile query_file(root / "data/stage_a/queries_u64x6.bin");
    const std::uint64_t rows = id_file.bytes() / sizeof(std::uint64_t);
    if (fp_file.bytes() != rows * WORDS * sizeof(std::uint64_t) ||
        pc_file.bytes() != rows * sizeof(std::uint16_t))
      throw std::runtime_error("union component-size mismatch");
    const auto *fp = fp_file.as<std::uint64_t>();
    const auto *ids = id_file.as<std::uint64_t>();
    const auto *pc = pc_file.as<std::uint16_t>();
    std::array<std::uint64_t, WORDS * 64 + 2> bins{};
    std::uint16_t previous = 0;
    for (std::uint64_t row = 0; row < rows; ++row) {
      if (pc[row] > WORDS * 64 || (row && pc[row] < previous))
        throw std::runtime_error("union popcount order mismatch");
      bins[pc[row] + 1]++;
      previous = pc[row];
    }
    for (std::size_t index = 1; index < bins.size(); ++index)
      bins[index] += bins[index - 1];
    const auto queries = load_queries(query_file, query_limit);
    const std::array<std::pair<int, int>, 2> thresholds{{{7, 10}, {4, 5}}};
    std::vector<Record> records(queries.size() * thresholds.size());

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t task = 0;
         task < static_cast<std::int64_t>(records.size()); ++task) {
      const auto begin_time = std::chrono::steady_clock::now();
      const int query_index = static_cast<int>(task / thresholds.size());
      const int threshold_index = static_cast<int>(task % thresholds.size());
      const auto [num, den] = thresholds[threshold_index];
      const auto &query = queries[query_index];
      const int lower = (num * static_cast<int>(query.popcount) + den - 1) / den;
      const int upper = den * static_cast<int>(query.popcount) / num;
      const std::uint64_t first = bins[std::max(0, lower)];
      const std::uint64_t last = bins[std::min(WORDS * 64, upper) + 1];
      std::vector<std::uint64_t> hit_ids;
      for (std::uint64_t row = first; row < last; ++row) {
        unsigned intersection = 0;
        for (int word = 0; word < WORDS; ++word)
          intersection +=
              __builtin_popcountll(query.fp[word] & fp[row * WORDS + word]);
        unsigned union_count = static_cast<unsigned>(query.popcount) + pc[row] -
                               intersection;
        if (union_count == 0) union_count = 1;
        if (intersection * static_cast<unsigned>(den) >=
            union_count * static_cast<unsigned>(num))
          hit_ids.push_back(ids[row]);
      }
      std::sort(hit_ids.begin(), hit_ids.end());
      const bool duplicate =
          std::adjacent_find(hit_ids.begin(), hit_ids.end()) != hit_ids.end();
      records[task] = Record{
          query_index,
          num,
          den,
          last - first,
          hit_ids.size(),
          hash_ids(hit_ids),
          duplicate,
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        begin_time)
              .count()};
    }

    if (!output_path.parent_path().empty())
      std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    output << "query,threshold_num,threshold_den,candidate_rows,hits,id_hash,"
              "duplicate_id,diagnostic_cpu_seconds\n";
    std::uint64_t duplicates = 0;
    for (const auto &record : records) {
      duplicates += record.duplicate_id;
      output << record.query << ',' << record.numerator << ','
             << record.denominator << ',' << record.candidates << ','
             << record.hits << ',' << record.id_hash << ','
             << record.duplicate_id << ',' << std::setprecision(17)
             << record.seconds << '\n';
    }
    std::ofstream summary(output_path.string() + ".summary.json");
    summary << "{\n"
            << "  \"experiment_id\": \"tide_20260829_gate6_exact_id_oracle\",\n"
            << "  \"rows\": " << rows << ",\n"
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
