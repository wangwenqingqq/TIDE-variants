// Independent CPU-only exact oracle for the Safe-C2 v4 frozen fresh SIFT-learn workload.
//
// It intentionally has no CUDA, GTS, gamma, timing, or traversal dependency.
// It evaluates every frozen query against the fixed SIFT1M base using:
//   (1) int64 squared L2 on verified integral float coordinates; and
//   (2) ordered float32 accumulation plus float32 sqrt.
// The output is an audit artifact for deterministic tie admission only.
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <unistd.h>

namespace fs = std::filesystem;
namespace {

constexpr int kDim = 128;
constexpr int kBaseN = 1'000'000;
constexpr int kQueryN = 12'524;
constexpr int kTop = 101;
constexpr int kGtWidth = 100;
constexpr std::uint64_t kAuditMagic = 0x31564f3443545347ULL;  // "GTS C4OV1"
constexpr std::int32_t kAuditVersion = 1;

[[noreturn]] void die(const std::string& message) {
  throw std::runtime_error("C2_V4_CPU_ORACLE: " + message);
}

struct Candidate {
  std::int32_t id = -1;
  std::int64_t d2 = std::numeric_limits<std::int64_t>::max();
  float fp = std::numeric_limits<float>::infinity();
};

struct ExactWorse {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.d2 != b.d2) return a.d2 < b.d2;
    return a.id < b.id;
  }
};

struct FpWorse {
  bool operator()(const Candidate& a, const Candidate& b) const {
    if (a.fp != b.fp) return a.fp < b.fp;
    return a.id < b.id;
  }
};

bool exact_better(const Candidate& a, const Candidate& b) {
  return a.d2 != b.d2 ? a.d2 < b.d2 : a.id < b.id;
}

bool fp_better(const Candidate& a, const Candidate& b) {
  return a.fp != b.fp ? a.fp < b.fp : a.id < b.id;
}

template <typename Heap, typename Better>
void consider(Heap& heap, const Candidate& candidate, Better better) {
  if (static_cast<int>(heap.size()) < kTop) {
    heap.push(candidate);
    return;
  }
  if (better(candidate, heap.top())) {
    heap.pop();
    heap.push(candidate);
  }
}

template <typename Heap, typename Better>
std::array<Candidate, kTop> sorted_heap(Heap heap, Better better) {
  if (static_cast<int>(heap.size()) != kTop) die("top heap cardinality mismatch");
  std::array<Candidate, kTop> out{};
  for (int i = 0; i < kTop; ++i) {
    out[i] = heap.top();
    heap.pop();
  }
  std::sort(out.begin(), out.end(), better);
  return out;
}

struct QueryResult {
  std::array<Candidate, kTop> exact{};
  std::array<Candidate, kTop> fp{};
  bool exact_boundary_tie = false;
  bool fp_boundary_tie = false;
  bool top10_idset_agree = false;
  bool fp_top101_collision_different_d2 = false;
  bool eligible = false;
};

std::uint32_t f32_bits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

void finalize(QueryResult* result) {
  result->exact_boundary_tie = result->exact[9].d2 == result->exact[10].d2;
  result->fp_boundary_tie = result->fp[9].fp == result->fp[10].fp;

  std::array<int, 10> exact_ids{};
  std::array<int, 10> fp_ids{};
  for (int i = 0; i < 10; ++i) {
    exact_ids[i] = result->exact[i].id;
    fp_ids[i] = result->fp[i].id;
  }
  std::sort(exact_ids.begin(), exact_ids.end());
  std::sort(fp_ids.begin(), fp_ids.end());
  result->top10_idset_agree = exact_ids == fp_ids;

  for (int begin = 0; begin < kTop;) {
    int end = begin + 1;
    while (end < kTop && result->fp[end].fp == result->fp[begin].fp) ++end;
    if (end - begin > 1) {
      const std::int64_t d2 = result->fp[begin].d2;
      for (int i = begin + 1; i < end; ++i) {
        if (result->fp[i].d2 != d2) result->fp_top101_collision_different_d2 = true;
      }
    }
    begin = end;
  }
  result->eligible = !result->exact_boundary_tie && !result->fp_boundary_tie &&
                     result->top10_idset_agree && !result->fp_top101_collision_different_d2;
}

void require_direct_regular_file(const fs::path& path, const std::string& label) {
  if (!path.is_absolute()) die(label + " must be an absolute path");
  std::error_code ec;
  const fs::file_status linked = fs::symlink_status(path, ec);
  if (ec || !fs::is_regular_file(linked)) die(label + " must be a direct regular file");
  const fs::path canonical = fs::canonical(path, ec);
  if (ec || canonical != path) die(label + " must not traverse a symlink");
}

void require_direct_directory(const fs::path& path, const std::string& label) {
  if (!path.is_absolute()) die(label + " must be an absolute path");
  std::error_code ec;
  const fs::file_status linked = fs::symlink_status(path, ec);
  if (ec || !fs::is_directory(linked)) die(label + " must be a direct directory");
  const fs::path canonical = fs::canonical(path, ec);
  if (ec || canonical != path) die(label + " must not traverse a symlink");
}

void require_absent(const fs::path& path, const std::string& label) {
  std::error_code ec;
  const fs::file_status linked = fs::symlink_status(path, ec);
  if (!ec && linked.type() != fs::file_type::not_found) die(label + " already exists or is a symlink");
  if (ec && ec != std::errc::no_such_file_or_directory) die("cannot inspect " + label);
}

std::vector<float> read_fvecs(const fs::path& path, int expected_rows, const std::string& label) {
  require_direct_regular_file(path, label);
  constexpr std::uintmax_t kRecordBytes = static_cast<std::uintmax_t>(sizeof(std::int32_t)) +
                                           static_cast<std::uintmax_t>(kDim) * sizeof(float);
  std::error_code ec;
  const std::uintmax_t expected_bytes = static_cast<std::uintmax_t>(expected_rows) * kRecordBytes;
  if (fs::file_size(path, ec) != expected_bytes || ec) die(label + " size/layout mismatch");

  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open " + label);
  std::vector<float> values(static_cast<std::size_t>(expected_rows) * kDim);
  for (int row = 0; row < expected_rows; ++row) {
    std::int32_t dim = 0;
    in.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    if (!in || dim != kDim) die(label + " dimension mismatch at row " + std::to_string(row));
    float* dst = values.data() + static_cast<std::size_t>(row) * kDim;
    in.read(reinterpret_cast<char*>(dst), static_cast<std::streamsize>(kDim * sizeof(float)));
    if (!in) die(label + " truncated at row " + std::to_string(row));
    for (int d = 0; d < kDim; ++d) {
      const float value = dst[d];
      if (!std::isfinite(value) || std::floor(value) != value || std::fabs(value) > 1.0e6F) {
        die(label + " has non-finite/non-integral/out-of-range coordinate at row " + std::to_string(row));
      }
    }
  }
  char extra = 0;
  if (in.read(&extra, 1)) die(label + " has trailing bytes");
  return values;
}

Candidate distance_to(const float* base, const float* query, int id) {
  std::int64_t d2 = 0;
  volatile float fp_sum = 0.0F;  // force ordered float32 accumulation, no reassociation.
  for (int d = 0; d < kDim; ++d) {
    const int base_i = static_cast<int>(base[d]);
    const int query_i = static_cast<int>(query[d]);
    const int delta_i = base_i - query_i;
    d2 += static_cast<std::int64_t>(delta_i) * static_cast<std::int64_t>(delta_i);
    const float delta = base[d] - query[d];
    const float term = delta * delta;
    fp_sum = fp_sum + term;
  }
  const float fp = std::sqrt(static_cast<float>(fp_sum));
  if (!std::isfinite(fp)) die("non-finite float32 distance");
  return Candidate{static_cast<std::int32_t>(id), d2, fp};
}

void write_i32(std::ofstream* out, std::int32_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*out) die("failed writing int32");
}

void write_i64(std::ofstream* out, std::int64_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*out) die("failed writing int64");
}

void write_u32(std::ofstream* out, std::uint32_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*out) die("failed writing uint32");
}

void write_u64(std::ofstream* out, std::uint64_t value) {
  out->write(reinterpret_cast<const char*>(&value), sizeof(value));
  if (!*out) die("failed writing uint64");
}

void close_or_die(std::ofstream* out, const std::string& label) {
  out->close();
  if (!*out) die("failed closing " + label);
}

struct Args {
  fs::path base;
  fs::path query;
  fs::path out;
  bool self_test = false;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    if (key == "--self-test") {
      if (args.self_test) die("duplicate --self-test");
      args.self_test = true;
      continue;
    }
    if (key == "--help") {
      std::cout << "Usage: c2_v4_cpu_exact_oracle --base ABS_BASE.fvecs --query ABS_FRESH12524.fvecs --out ABS_NEW_DIR\n"
                << "       c2_v4_cpu_exact_oracle --self-test\n";
      std::exit(0);
    }
    if (++i >= argc) die("missing value after " + key);
    const fs::path value(argv[i]);
    if (key == "--base") {
      if (!args.base.empty()) die("duplicate --base");
      args.base = value;
    } else if (key == "--query") {
      if (!args.query.empty()) die("duplicate --query");
      args.query = value;
    } else if (key == "--out") {
      if (!args.out.empty()) die("duplicate --out");
      args.out = value;
    } else {
      die("unknown argument " + key);
    }
  }
  if (args.self_test) {
    if (!args.base.empty() || !args.query.empty() || !args.out.empty()) {
      die("--self-test accepts no other arguments");
    }
    return args;
  }
  if (args.base.empty() || args.query.empty() || args.out.empty()) die("--base --query --out are required");
  return args;
}

void run_self_test() {
  std::vector<float> base(static_cast<std::size_t>(kTop) * kDim, 0.0F);
  std::array<float, kDim> query{};
  for (int i = 0; i < kTop; ++i) base[static_cast<std::size_t>(i) * kDim] = static_cast<float>(i + 1);

  std::priority_queue<Candidate, std::vector<Candidate>, ExactWorse> exact_heap;
  std::priority_queue<Candidate, std::vector<Candidate>, FpWorse> fp_heap;
  for (int i = 0; i < kTop; ++i) {
    const Candidate candidate = distance_to(base.data() + static_cast<std::size_t>(i) * kDim, query.data(), i);
    consider(exact_heap, candidate, exact_better);
    consider(fp_heap, candidate, fp_better);
  }
  QueryResult clean;
  clean.exact = sorted_heap(exact_heap, exact_better);
  clean.fp = sorted_heap(fp_heap, fp_better);
  finalize(&clean);
  if (!clean.eligible) die("self-test expected eligible unique-distance query");

  QueryResult tied = clean;
  tied.exact[10].d2 = tied.exact[9].d2;
  tied.fp[10].fp = tied.fp[9].fp;
  finalize(&tied);
  if (tied.eligible || !tied.exact_boundary_tie || !tied.fp_boundary_tie) {
    die("self-test tie admission did not fail closed");
  }
  std::cout << R"JSON({"status":"PASS_C2_V4_CPU_ORACLE_SELF_TEST","gpu_binary_executed":false,"nvidia_smi_called":false,"unique_case_eligible":true,"boundary_tie_rejected":true})JSON" << '\n';
}

void write_outputs(const fs::path& out, const std::vector<QueryResult>& all) {
  std::ofstream gt(out / "groundtruth_fp32_top100.ivecs", std::ios::binary);
  std::ofstream audit(out / "top101_int64_fp32_audit.bin", std::ios::binary);
  std::ofstream admission(out / "tie_admission.tsv");
  std::ofstream summary(out / "oracle_summary.json");
  if (!gt || !audit || !admission || !summary) die("cannot create oracle output files");

  write_u64(&audit, kAuditMagic);
  write_i32(&audit, kAuditVersion);
  write_i32(&audit, kDim);
  write_i32(&audit, kBaseN);
  write_i32(&audit, kQueryN);
  write_i32(&audit, kTop);
  write_i32(&audit, kGtWidth);
  admission << "local_id\texact_rank10_d2\texact_rank11_d2\tfp32_rank10_bits\tfp32_rank11_bits\t"
            << "exact_boundary_tie\tfp32_boundary_tie\ttop10_idset_agree\t"
            << "fp32_top101_collision_different_d2\teligible\n";

  int eligible = 0;
  int exact_ties = 0;
  int fp_ties = 0;
  int top10_mismatch = 0;
  int fp_collision = 0;
  for (int qid = 0; qid < kQueryN; ++qid) {
    const QueryResult& result = all[static_cast<std::size_t>(qid)];
    write_i32(&gt, kGtWidth);
    for (int rank = 0; rank < kGtWidth; ++rank) write_i32(&gt, result.fp[rank].id);

    write_i32(&audit, qid);
    for (int rank = 0; rank < kTop; ++rank) {
      const Candidate& exact = result.exact[rank];
      const Candidate& fp = result.fp[rank];
      write_i32(&audit, exact.id);
      write_i64(&audit, exact.d2);
      write_u32(&audit, f32_bits(exact.fp));
      write_i32(&audit, fp.id);
      write_i64(&audit, fp.d2);
      write_u32(&audit, f32_bits(fp.fp));
    }

    admission << qid << '\t' << result.exact[9].d2 << '\t' << result.exact[10].d2 << '\t'
              << f32_bits(result.fp[9].fp) << '\t' << f32_bits(result.fp[10].fp) << '\t'
              << static_cast<int>(result.exact_boundary_tie) << '\t'
              << static_cast<int>(result.fp_boundary_tie) << '\t'
              << static_cast<int>(result.top10_idset_agree) << '\t'
              << static_cast<int>(result.fp_top101_collision_different_d2) << '\t'
              << static_cast<int>(result.eligible) << '\n';
    eligible += result.eligible ? 1 : 0;
    exact_ties += result.exact_boundary_tie ? 1 : 0;
    fp_ties += result.fp_boundary_tie ? 1 : 0;
    top10_mismatch += result.top10_idset_agree ? 0 : 1;
    fp_collision += result.fp_top101_collision_different_d2 ? 1 : 0;
  }

  summary << "{\n"
          << "  \"schema\": \"safe-c2-v4-cpu-exact-fp32-int64-oracle-v1\",\n"
          << "  \"status\": \"COMPLETE_CPU_ONLY_ORACLE_UNADMITTED\",\n"
          << "  \"dimension\": " << kDim << ",\n"
          << "  \"base_count\": " << kBaseN << ",\n"
          << "  \"query_count\": " << kQueryN << ",\n"
          << "  \"top_width\": " << kTop << ",\n"
          << "  \"groundtruth_width\": " << kGtWidth << ",\n"
          << "  \"int64_semantics\": \"sum of squared differences after verified integral float32 coordinates\",\n"
          << "  \"fp32_semantics\": \"ordered volatile float32 sum of 128 delta*delta terms, then float32 sqrt; compiled with fp contraction disabled\",\n"
          << "  \"tie_admission_rule\": \"reject exact rank10==rank11, fp32 rank10==rank11, exact/fp32 top10 ID-set mismatch, or any equal-fp32/different-int64 collision in top101; no replacement\",\n"
          << "  \"eligible_total\": " << eligible << ",\n"
          << "  \"exact_boundary_ties\": " << exact_ties << ",\n"
          << "  \"fp32_boundary_ties\": " << fp_ties << ",\n"
          << "  \"top10_idset_mismatches\": " << top10_mismatch << ",\n"
          << "  \"fp32_top101_collisions_different_d2\": " << fp_collision << ",\n"
          << "  \"gpu_binary_executed\": false,\n"
          << "  \"nvidia_smi_called\": false\n"
          << "}\n";

  close_or_die(&gt, "groundtruth");
  close_or_die(&audit, "top101 audit");
  close_or_die(&admission, "tie admission");
  close_or_die(&summary, "summary");
}

void run_oracle(const Args& args) {
  require_direct_regular_file(args.base, "base fvecs");
  require_direct_regular_file(args.query, "fresh query fvecs");
  if (!args.out.is_absolute()) die("output directory must be an absolute path");
  require_direct_directory(args.out.parent_path(), "output parent");
  require_absent(args.out, "oracle output directory");

  const fs::path temp = args.out.parent_path() /
      (args.out.filename().string() + ".tmp_cpu_oracle_" + std::to_string(static_cast<long long>(::getpid())));
  require_absent(temp, "oracle temporary directory");
  if (!fs::create_directory(temp)) die("cannot create oracle temporary directory");
  bool committed = false;
  try {
    const std::vector<float> base = read_fvecs(args.base, kBaseN, "base fvecs");
    const std::vector<float> queries = read_fvecs(args.query, kQueryN, "fresh query fvecs");
    std::vector<QueryResult> all(static_cast<std::size_t>(kQueryN));
    std::atomic<int> completed{0};

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
    for (int qid = 0; qid < kQueryN; ++qid) {
      std::priority_queue<Candidate, std::vector<Candidate>, ExactWorse> exact_heap;
      std::priority_queue<Candidate, std::vector<Candidate>, FpWorse> fp_heap;
      const float* query = queries.data() + static_cast<std::size_t>(qid) * kDim;
      for (int id = 0; id < kBaseN; ++id) {
        const Candidate candidate = distance_to(base.data() + static_cast<std::size_t>(id) * kDim, query, id);
        consider(exact_heap, candidate, exact_better);
        consider(fp_heap, candidate, fp_better);
      }
      QueryResult& result = all[static_cast<std::size_t>(qid)];
      result.exact = sorted_heap(exact_heap, exact_better);
      result.fp = sorted_heap(fp_heap, fp_better);
      finalize(&result);
      const int done = ++completed;
      if (done % 100 == 0 || done == kQueryN) {
#ifdef _OPENMP
#pragma omp critical
#endif
        std::cerr << "C2_V4_CPU_ORACLE_PROGRESS completed_queries=" << done << "/" << kQueryN << "\n";
      }
    }

    write_outputs(temp, all);
    fs::rename(temp, args.out);
    committed = true;
    std::cout << "C2_V4_CPU_ORACLE_COMPLETE output=" << args.out.string()
              << " query_count=" << kQueryN << "\n";
  } catch (...) {
    if (!committed) {
      std::error_code cleanup_ec;
      fs::remove_all(temp, cleanup_ec);
    }
    throw;
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    if (args.self_test) run_self_test();
    else run_oracle(args);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
