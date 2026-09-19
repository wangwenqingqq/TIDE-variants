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
  MappedFile(const MappedFile &) = delete;
  MappedFile &operator=(const MappedFile &) = delete;

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
  std::array<std::uint64_t, WORDS> fp{};
  std::uint16_t popcount = 0;
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

class Database {
 public:
  Database(const std::filesystem::path &root, const std::string &prefix)
      : fp_file_(root / "data/gate0_prepared" /
                 (prefix + "_fp_u64x4.bin")),
        id_file_(root / "data/gate0_prepared" /
                 (prefix + "_id_i64.bin")),
        pc_file_(root / "data/gate0_prepared" /
                 (prefix + "_popcnt_u16.bin")),
        fp_(fp_file_.as<std::uint64_t>()),
        ids_(id_file_.as<std::uint64_t>()),
        pc_(pc_file_.as<std::uint16_t>()) {
    rows_ = id_file_.bytes() / sizeof(std::uint64_t);
    if (fp_file_.bytes() != rows_ * WORDS * sizeof(std::uint64_t) ||
        pc_file_.bytes() != rows_ * sizeof(std::uint16_t))
      throw std::runtime_error(prefix + " component-size mismatch");
    std::uint16_t previous = 0;
    for (std::uint64_t row = 0; row < rows_; ++row) {
      unsigned actual = 0;
      for (int word = 0; word < WORDS; ++word)
        actual += __builtin_popcountll(fp_[row * WORDS + word]);
      if (pc_[row] != actual || pc_[row] > WORDS * 64 ||
          (row && pc_[row] < previous))
        throw std::runtime_error(prefix + " popcount/order mismatch");
      bins_[pc_[row] + 1]++;
      previous = pc_[row];
    }
    for (std::size_t index = 1; index < bins_.size(); ++index)
      bins_[index] += bins_[index - 1];
  }

  std::uint64_t rows() const { return rows_; }
  const std::uint64_t *fp() const { return fp_; }
  const std::uint64_t *ids() const { return ids_; }
  const std::uint16_t *pc() const { return pc_; }
  const std::array<std::uint64_t, WORDS * 64 + 2> &bins() const {
    return bins_;
  }

 private:
  MappedFile fp_file_;
  MappedFile id_file_;
  MappedFile pc_file_;
  const std::uint64_t *fp_ = nullptr;
  const std::uint64_t *ids_ = nullptr;
  const std::uint16_t *pc_ = nullptr;
  std::uint64_t rows_ = 0;
  std::array<std::uint64_t, WORDS * 64 + 2> bins_{};
};

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
  int epoch = 0;
  int query = 0;
  int numerator = 0;
  int denominator = 0;
  std::uint64_t candidates = 0;
  std::uint64_t hits = 0;
  std::uint64_t id_hash = 0;
  bool contains_query_id = false;
  bool duplicate_id = false;
  double seconds = 0;
};

Record scan(int epoch, const Database &database, const Query &query,
            int query_index, int num, int den) {
  const auto begin_time = std::chrono::steady_clock::now();
  const int lower = (num * static_cast<int>(query.popcount) + den - 1) / den;
  const int upper = den * static_cast<int>(query.popcount) / num;
  const auto &bins = database.bins();
  const std::uint64_t first = bins[std::max(0, lower)];
  const std::uint64_t last = bins[std::min(WORDS * 64, upper) + 1];
  std::vector<std::uint64_t> hit_ids;
  for (std::uint64_t row = first; row < last; ++row) {
    unsigned intersection = 0;
    for (int word = 0; word < WORDS; ++word)
      intersection += __builtin_popcountll(
          query.fp[word] & database.fp()[row * WORDS + word]);
    unsigned union_count = static_cast<unsigned>(query.popcount) +
                           database.pc()[row] - intersection;
    if (union_count == 0) union_count = 1;
    if (intersection * static_cast<unsigned>(den) >=
        union_count * static_cast<unsigned>(num))
      hit_ids.push_back(database.ids()[row]);
  }
  std::sort(hit_ids.begin(), hit_ids.end());
  return Record{
      epoch,
      query_index,
      num,
      den,
      last - first,
      hit_ids.size(),
      hash_ids(hit_ids),
      std::binary_search(hit_ids.begin(), hit_ids.end(), query.id),
      std::adjacent_find(hit_ids.begin(), hit_ids.end()) != hit_ids.end(),
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    begin_time)
          .count()};
}

}  // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      std::cerr << "usage: " << argv[0]
                << " TRANSITION_ROOT OUTPUT_CSV QUERY_LIMIT\n";
      return 64;
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path output_path = argv[2];
    const int query_limit = std::stoi(argv[3]);
    if (query_limit <= 0) throw std::runtime_error("QUERY_LIMIT must be positive");

    Database base(root, "base");
    Database fresh(root, "union");
    if (base.rows() >= fresh.rows())
      throw std::runtime_error("expected a non-empty insert-only delta");
    MappedFile query_file(root / "data/stage_a/queries_u64x6.bin");
    const auto queries = load_queries(query_file, query_limit);
    const std::array<const Database *, 2> databases{{&base, &fresh}};
    const std::array<std::pair<int, int>, 2> thresholds{{{7, 10}, {4, 5}}};
    const std::size_t per_epoch = queries.size() * thresholds.size();
    std::vector<Record> records(databases.size() * per_epoch);

#pragma omp parallel for schedule(dynamic)
    for (std::int64_t task = 0;
         task < static_cast<std::int64_t>(records.size()); ++task) {
      const int epoch = static_cast<int>(task / per_epoch);
      const int within = static_cast<int>(task % per_epoch);
      const int query_index = within / static_cast<int>(thresholds.size());
      const int threshold_index = within % static_cast<int>(thresholds.size());
      const auto [num, den] = thresholds[threshold_index];
      records[task] = scan(epoch, *databases[epoch], queries[query_index],
                           query_index, num, den);
    }

    if (!output_path.parent_path().empty())
      std::filesystem::create_directories(output_path.parent_path());
    std::ofstream output(output_path);
    output << "epoch,query,threshold_num,threshold_den,candidate_rows,hits,"
              "id_hash,contains_query_id,duplicate_id,diagnostic_cpu_seconds\n";
    std::uint64_t duplicates = 0;
    std::uint64_t base_contains = 0;
    std::uint64_t union_missing = 0;
    for (const auto &record : records) {
      duplicates += record.duplicate_id;
      base_contains += record.epoch == 0 && record.contains_query_id;
      union_missing += record.epoch == 1 && !record.contains_query_id;
      output << record.epoch << ',' << record.query << ',' << record.numerator
             << ',' << record.denominator << ',' << record.candidates << ','
             << record.hits << ',' << record.id_hash << ','
             << record.contains_query_id << ',' << record.duplicate_id << ','
             << std::setprecision(17) << record.seconds << '\n';
    }
    const bool pass = duplicates == 0 && base_contains == 0 && union_missing == 0;
    std::ofstream summary(output_path.string() + ".summary.json");
    summary << "{\n"
            << "  \"experiment_id\": "
               "\"tide_20260903_read_while_update_epoch_oracle\",\n"
            << "  \"base_rows\": " << base.rows() << ",\n"
            << "  \"union_rows\": " << fresh.rows() << ",\n"
            << "  \"queries\": " << queries.size() << ",\n"
            << "  \"requests_per_epoch\": " << per_epoch << ",\n"
            << "  \"total_records\": " << records.size() << ",\n"
            << "  \"duplicate_id_records\": " << duplicates << ",\n"
            << "  \"base_contains_future_query_records\": " << base_contains
            << ",\n"
            << "  \"union_missing_future_query_records\": " << union_missing
            << ",\n"
            << "  \"pass\": " << (pass ? "true" : "false") << "\n}\n";
    return pass ? 0 : 2;
  } catch (const std::exception &error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
