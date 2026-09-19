#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr int K = 10;

struct Candidate {
  std::int64_t id;
  std::uint32_t row;
  std::uint16_t num;
  std::uint16_t den;
};

inline Candidate sentinel() { return Candidate{INT64_MAX, UINT32_MAX, 0, 1}; }

inline bool better(const Candidate &a, const Candidate &b) {
  const std::uint32_t lhs = static_cast<std::uint32_t>(a.num) * b.den;
  const std::uint32_t rhs = static_cast<std::uint32_t>(b.num) * a.den;
  if (lhs != rhs) return lhs > rhs;
  if (a.id != b.id) return a.id < b.id;
  return a.row < b.row;
}

struct Top10 {
  Candidate v[K];
  void init() { for (auto &x : v) x = sentinel(); }
  inline void insert(const Candidate c) {
    if (!better(c, v[K - 1])) return;
    int pos = K - 1;
    while (pos > 0 && better(c, v[pos - 1])) {
      v[pos] = v[pos - 1];
      --pos;
    }
    v[pos] = c;
  }
};

class MappedFile {
 public:
  explicit MappedFile(const std::string &path) : path_(path) {
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("open failed: " + path);
    struct stat st {};
    if (fstat(fd_, &st) != 0) throw std::runtime_error("fstat failed: " + path);
    bytes_ = static_cast<std::size_t>(st.st_size);
    data_ = mmap(nullptr, bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (data_ == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  }
  ~MappedFile() {
    if (data_ != MAP_FAILED) munmap(data_, bytes_);
    if (fd_ >= 0) close(fd_);
  }
  template <class T> const T *as() const {
    if (bytes_ % sizeof(T)) throw std::runtime_error("bad size: " + path_);
    return static_cast<const T *>(data_);
  }
  std::size_t bytes() const { return bytes_; }

 private:
  std::string path_;
  int fd_ = -1;
  std::size_t bytes_ = 0;
  void *data_ = MAP_FAILED;
};

struct Options {
  std::string root, dataset = "union", output;
  int query_offset = 0, query_count = 0, threads = 1;
};

Options parse_args(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto need = [&]() {
      if (++i >= argc) throw std::runtime_error("missing value for " + a);
      return std::string(argv[i]);
    };
    if (a == "--root") o.root = need();
    else if (a == "--dataset") o.dataset = need();
    else if (a == "--query-offset") o.query_offset = std::stoi(need());
    else if (a == "--query-count") o.query_count = std::stoi(need());
    else if (a == "--threads") o.threads = std::stoi(need());
    else if (a == "--output") o.output = need();
    else throw std::runtime_error("unknown argument: " + a);
  }
  if (o.root.empty() || o.output.empty() || o.query_count < 1 || o.threads < 1)
    throw std::runtime_error("missing required arguments");
  return o;
}
}  // namespace

int main(int argc, char **argv) {
  try {
    const Options o = parse_args(argc, argv);
    const std::string p = o.root + "/data/prepared/";
    MappedFile fp_file(p + o.dataset + "_fp_u64x4.bin");
    MappedFile id_file(p + o.dataset + "_id_i64.bin");
    MappedFile pc_file(p + o.dataset + "_popcnt_u16.bin");
    MappedFile qfp_file(p + "queries_fp_u64x4.bin");
    MappedFile qid_file(p + "queries_id_i64.bin");
    MappedFile qpc_file(p + "queries_popcnt_u16.bin");
    const std::uint64_t rows = id_file.bytes() / sizeof(std::int64_t);
    const std::uint64_t qrows = qid_file.bytes() / sizeof(std::int64_t);
    if (fp_file.bytes() != rows * 4 * sizeof(std::uint64_t) ||
        pc_file.bytes() != rows * sizeof(std::uint16_t) ||
        qfp_file.bytes() != qrows * 4 * sizeof(std::uint64_t) ||
        qpc_file.bytes() != qrows * sizeof(std::uint16_t) ||
        o.query_offset < 0 ||
        static_cast<std::uint64_t>(o.query_offset + o.query_count) > qrows)
      throw std::runtime_error("file size or query range mismatch");

    const auto *fp = fp_file.as<std::uint64_t>();
    const auto *ids = id_file.as<std::int64_t>();
    const auto *pc = pc_file.as<std::uint16_t>();
    const auto *qfp = qfp_file.as<std::uint64_t>() +
                      static_cast<std::uint64_t>(o.query_offset) * 4;
    const auto *qids = qid_file.as<std::int64_t>() + o.query_offset;
    const auto *qpc = qpc_file.as<std::uint16_t>() + o.query_offset;
    std::vector<Candidate> out(static_cast<std::size_t>(o.query_count) * K);

#ifdef _OPENMP
    omp_set_num_threads(o.threads);
#endif
    const auto start = std::chrono::steady_clock::now();
#pragma omp parallel for schedule(static)
    for (int q = 0; q < o.query_count; ++q) {
      const std::uint64_t q0 = qfp[static_cast<std::uint64_t>(q) * 4 + 0];
      const std::uint64_t q1 = qfp[static_cast<std::uint64_t>(q) * 4 + 1];
      const std::uint64_t q2 = qfp[static_cast<std::uint64_t>(q) * 4 + 2];
      const std::uint64_t q3 = qfp[static_cast<std::uint64_t>(q) * 4 + 3];
      Top10 top; top.init();
      for (std::uint64_t row = 0; row < rows; ++row) {
        if (ids[row] == qids[q]) continue;
        const auto off = row * 4;
        const unsigned inter =
            __builtin_popcountll(q0 & fp[off + 0]) +
            __builtin_popcountll(q1 & fp[off + 1]) +
            __builtin_popcountll(q2 & fp[off + 2]) +
            __builtin_popcountll(q3 & fp[off + 3]);
        unsigned uni = static_cast<unsigned>(qpc[q]) + pc[row] - inter;
        if (uni == 0) uni = 1;
        top.insert(Candidate{ids[row], static_cast<std::uint32_t>(row),
                             static_cast<std::uint16_t>(inter),
                             static_cast<std::uint16_t>(uni)});
      }
      std::copy(top.v, top.v + K, out.begin() + static_cast<std::size_t>(q) * K);
    }
    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();

    std::ofstream csv(o.output);
    csv << "query_index,query_id,rank,result_id,row,num,den,similarity\n"
        << std::setprecision(12);
    for (int q = 0; q < o.query_count; ++q)
      for (int k = 0; k < K; ++k) {
        const auto &c = out[q * K + k];
        csv << (o.query_offset + q) << ',' << qids[q] << ',' << k << ',' << c.id
            << ',' << c.row << ',' << c.num << ',' << c.den << ','
            << static_cast<double>(c.num) / c.den << '\n';
      }
    std::cout << "dataset=" << o.dataset << "\nrows=" << rows
              << "\nquery_offset=" << o.query_offset
              << "\nquery_count=" << o.query_count << "\nthreads=" << o.threads
              << "\nscan_seconds=" << seconds << "\noutput=" << o.output << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << '\n';
    return 1;
  }
}
