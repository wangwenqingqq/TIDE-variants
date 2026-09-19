// Isolated full-SIFT1M submitted-VLDB C2 (per-level calibrated pruning) gate.
//
// Scope boundary:
//   * C2 only: static GTS per-level gamma calibration/evaluation.
//   * No update code, no incremental insert, no range exporter, no dynamic exact
//     smoke.  Neither original GTS archive is written.
//   * C2 is empirical recall-controlled pruning, NOT a lower-bound theorem.
//
// This file deliberately uses an isolated copy of the legacy headers under
// c2_gamma_heldout_scale/include.  `search_v2.cuh` has exactly one documented
// experiment-only change: a fixed 1 GiB workspace cap shared by every stage.

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "tree.cuh"
#include "search_v2.cuh"

// The archived headers declare this host routine but define it only in main.cu.
// Keep a local implementation to upload a fully explicit C2 gamma vector.
void upload_rp_constants(float* h_alpha, float* h_beta, float* h_gamma,
                         int /*num_levels*/, float* h_lut_breaks,
                         float* h_lut_slopes, float* h_lut_intercepts,
                         int lut_size, int mode) {
  CHECK(cudaMemcpyToSymbol(c_rp_alpha, h_alpha, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_beta, h_beta, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_gamma, h_gamma, RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  if (h_lut_breaks != nullptr && h_lut_slopes != nullptr && h_lut_intercepts != nullptr) {
    CHECK(cudaMemcpyToSymbol(c_lut_breaks, h_lut_breaks, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_slopes, h_lut_slopes, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_intercepts, h_lut_intercepts, RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  }
  CHECK(cudaMemcpyToSymbol(c_lut_num_segments, &lut_size, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
}

namespace c2 {

constexpr int kDimension = 128;
constexpr int kBaseN = 1'000'000;
constexpr int kQueryN = 10'000;
constexpr int kGtWidth = 100;
constexpr int kK = 10;
constexpr int kFreezeLevel = 2;  // levels 0..2 exact baseline
constexpr int kBinaryIterations = 16;
constexpr float kCandidateLo = 1.0F;
constexpr float kCandidateHi = 5.0F;
constexpr float kMu = 0.75F;
constexpr std::size_t kWorkspaceCapBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr float kDistanceAbsFloor = 2.0e-3F;
constexpr int kDistanceMaxUlps = 2;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("C2-PERLEVEL: " + message);
}

void cuda_or_die(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << "CUDA " << expression << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(status);
  fail(out.str());
}
#define C2_CUDA(expr) ::c2::cuda_or_die((expr), #expr, __FILE__, __LINE__)

struct Args {
  std::string mode;  // calibrate or evaluate
  std::string base_fvecs;
  std::string query_fvecs;
  std::string groundtruth_ivecs;
  std::string query_ids;
  std::string output_dir;
  std::string gamma_vector_file;
  std::string stage;
  int warmup_reps = 1;
  int timed_reps = 1;
  long long baseline_target_correct = -1;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto next = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return argv[i];
    };
    if (key == "--mode") args.mode = next();
    else if (key == "--base-fvecs") args.base_fvecs = next();
    else if (key == "--query-fvecs") args.query_fvecs = next();
    else if (key == "--groundtruth-ivecs") args.groundtruth_ivecs = next();
    else if (key == "--query-ids") args.query_ids = next();
    else if (key == "--out") args.output_dir = next();
    else if (key == "--gamma-vector-file") args.gamma_vector_file = next();
    else if (key == "--stage") args.stage = next();
    else if (key == "--warmup-reps") args.warmup_reps = std::stoi(next());
    else if (key == "--timed-reps") args.timed_reps = std::stoi(next());
    else if (key == "--baseline-target-correct") args.baseline_target_correct = std::stoll(next());
    else if (key == "--help") {
      std::cout
          << "usage: GTS_c2_perlevel_sift1m --mode calibrate|evaluate --base-fvecs BASE "
          << "--query-fvecs QUERY --groundtruth-ivecs GT --query-ids IDS --out DIR "
          << "--stage calibration|validation|test [--gamma-vector-file FILE] "
          << "[--baseline-target-correct N] [--warmup-reps 1] [--timed-reps N]\n";
      std::exit(0);
    } else {
      fail("unknown argument " + key);
    }
  }
  if (args.mode != "calibrate" && args.mode != "evaluate") fail("--mode must be calibrate or evaluate");
  if (args.base_fvecs.empty() || args.query_fvecs.empty() || args.groundtruth_ivecs.empty() ||
      args.query_ids.empty() || args.output_dir.empty() || args.stage.empty()) {
    fail("missing required input/output argument");
  }
  if (args.warmup_reps < 0 || args.timed_reps <= 0) fail("invalid repetition counts");
  if (args.mode == "calibrate") {
    if (args.stage != "calibration") fail("calibrate mode requires --stage calibration");
    if (!args.gamma_vector_file.empty() || args.baseline_target_correct != -1) {
      fail("calibrate mode derives its own all-ones baseline/vector");
    }
  } else {
    if (args.gamma_vector_file.empty() || args.baseline_target_correct < 0) {
      fail("evaluate mode requires --gamma-vector-file and --baseline-target-correct");
    }
    if (args.stage != "validation" && args.stage != "test") {
      fail("evaluate mode requires validation or test stage");
    }
  }
  return args;
}

struct Fvecs {
  int dimension = 0;
  int count = 0;
  std::vector<float> values;
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
};

Fvecs read_fvecs(const std::string& path, int required_count, int required_dimension) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open fvecs " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff bytes = input.tellg();
  input.seekg(0, std::ios::beg);
  int dimension = 0;
  input.read(reinterpret_cast<char*>(&dimension), sizeof(dimension));
  if (!input || dimension != required_dimension) fail("unexpected fvecs dimension in " + path);
  const std::streamoff record_bytes = static_cast<std::streamoff>((dimension + 1) * sizeof(float));
  if (bytes <= 0 || bytes % record_bytes != 0 || bytes / record_bytes != required_count) {
    fail("unexpected fvecs count/layout in " + path);
  }
  input.seekg(0, std::ios::beg);
  Fvecs out;
  out.dimension = dimension;
  out.count = required_count;
  out.values.resize(static_cast<std::size_t>(required_count) * dimension);
  for (int row = 0; row < required_count; ++row) {
    int row_dimension = 0;
    input.read(reinterpret_cast<char*>(&row_dimension), sizeof(row_dimension));
    if (!input || row_dimension != dimension) fail("inconsistent fvecs row header at " + std::to_string(row));
    float* destination = out.values.data() + static_cast<std::size_t>(row) * dimension;
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(dimension * sizeof(float)));
    if (!input) fail("truncated fvecs payload at " + std::to_string(row));
    for (int d = 0; d < dimension; ++d) {
      if (!std::isfinite(destination[d])) fail("non-finite fvecs value");
      out.min_value = std::min(out.min_value, destination[d]);
      out.max_value = std::max(out.max_value, destination[d]);
    }
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("extra fvecs payload");
  return out;
}

std::vector<int> read_ivecs(const std::string& path, int required_count, int required_width) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open ivecs " + path);
  input.seekg(0, std::ios::end);
  const std::streamoff bytes = input.tellg();
  input.seekg(0, std::ios::beg);
  int width = 0;
  input.read(reinterpret_cast<char*>(&width), sizeof(width));
  if (!input || width != required_width) fail("unexpected ivecs width");
  const std::streamoff record_bytes = static_cast<std::streamoff>((width + 1) * sizeof(int));
  if (bytes <= 0 || bytes % record_bytes != 0 || bytes / record_bytes != required_count) {
    fail("unexpected ivecs count/layout");
  }
  input.seekg(0, std::ios::beg);
  std::vector<int> out(static_cast<std::size_t>(required_count) * width);
  for (int row = 0; row < required_count; ++row) {
    int row_width = 0;
    input.read(reinterpret_cast<char*>(&row_width), sizeof(row_width));
    if (!input || row_width != width) fail("inconsistent ivecs row header at " + std::to_string(row));
    int* destination = out.data() + static_cast<std::size_t>(row) * width;
    input.read(reinterpret_cast<char*>(destination), static_cast<std::streamsize>(width * sizeof(int)));
    if (!input) fail("truncated ivecs payload at " + std::to_string(row));
    std::unordered_set<int> seen;
    for (int j = 0; j < width; ++j) {
      if (destination[j] < 0 || destination[j] >= kBaseN) fail("out-of-range GT id");
      if (!seen.insert(destination[j]).second) fail("duplicate GT id in row " + std::to_string(row));
    }
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("extra ivecs payload");
  return out;
}

std::vector<int> read_query_ids(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open query IDs " + path);
  std::vector<int> ids;
  std::unordered_set<int> seen;
  std::string line;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    std::size_t consumed = 0;
    int id = 0;
    try { id = std::stoi(line, &consumed); } catch (...) { fail("invalid query ID line"); }
    if (consumed != line.size() || id < 0 || id >= kQueryN || !seen.insert(id).second) {
      fail("invalid/duplicate query ID");
    }
    ids.push_back(id);
  }
  if (ids.empty()) fail("empty query ID list");
  return ids;
}

std::array<float, RP_MAX_LEVELS> read_gamma_vector(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("cannot open gamma vector " + path);
  std::array<float, RP_MAX_LEVELS> gamma{};
  std::string tag;
  for (int level = 0; level < RP_MAX_LEVELS; ++level) {
    int found_level = -1;
    if (!(input >> found_level >> gamma[level]) || found_level != level || !std::isfinite(gamma[level]) || gamma[level] < 1.0F) {
      fail("invalid gamma vector file");
    }
  }
  if (input >> tag) fail("extra gamma vector data");
  return gamma;
}

void write_gamma_vector(const std::string& path, const std::array<float, RP_MAX_LEVELS>& gamma) {
  std::ofstream out(path);
  if (!out) fail("cannot write gamma vector");
  out << std::setprecision(9);
  for (int level = 0; level < RP_MAX_LEVELS; ++level) out << level << ' ' << gamma[level] << '\n';
  if (!out) fail("failed writing gamma vector");
}

struct MetricPlan {
  long double bbox_squared = 0.0L;
  long double bbox_l2 = 0.0L;
  int infi_dis = 10000;
};

MetricPlan metric_plan(const Fvecs& base, const Fvecs& queries, const std::vector<int>& selected_ids) {
  std::vector<float> lo(kDimension, std::numeric_limits<float>::infinity());
  std::vector<float> hi(kDimension, -std::numeric_limits<float>::infinity());
  const auto absorb = [&](const float* vector) {
    for (int d = 0; d < kDimension; ++d) { lo[d] = std::min(lo[d], vector[d]); hi[d] = std::max(hi[d], vector[d]); }
  };
  for (int i = 0; i < base.count; ++i) absorb(base.values.data() + static_cast<std::size_t>(i) * kDimension);
  for (int qid : selected_ids) absorb(queries.values.data() + static_cast<std::size_t>(qid) * kDimension);
  MetricPlan plan;
  for (int d = 0; d < kDimension; ++d) { const long double delta = static_cast<long double>(hi[d]) - lo[d]; plan.bbox_squared += delta * delta; }
  plan.bbox_l2 = std::sqrt(plan.bbox_squared);
  const long double with_margin = std::ceil(plan.bbox_l2) + 1024.0L;
  if (with_margin > static_cast<long double>(std::numeric_limits<int>::max())) fail("infi_dis overflow");
  plan.infi_dis = std::max(10000, static_cast<int>(with_margin));
  return plan;
}

struct Runtime {
  int* data_info = nullptr;
  float* data_d = nullptr;
  int* id_list = nullptr;
  TN* node_list = nullptr;
  int* max_node_num = nullptr;
  int* empty_list = nullptr;
  int tree_height = 0;
};

void release_tree(Runtime& runtime) {
  if (runtime.id_list) { cudaFree(runtime.id_list); runtime.id_list = nullptr; }
  if (runtime.node_list) { cudaFree(runtime.node_list); runtime.node_list = nullptr; }
  if (runtime.max_node_num) { cudaFree(runtime.max_node_num); runtime.max_node_num = nullptr; }
  if (runtime.empty_list) { cudaFree(runtime.empty_list); runtime.empty_list = nullptr; }
  if (runtime.data_d) { cudaFree(runtime.data_d); runtime.data_d = nullptr; }
  if (runtime.data_info) { cudaFree(runtime.data_info); runtime.data_info = nullptr; }
  if (max_dis_d) { cudaFree(max_dis_d); max_dis_d = nullptr; }
  if (split_list) { cudaFree(split_list); split_list = nullptr; }
  if (pid_list) { cudaFree(pid_list); pid_list = nullptr; }
  if (dis_list) { cudaFree(dis_list); dis_list = nullptr; }
  if (split_num) { cudaFree(split_num); split_num = nullptr; }
}

std::uint64_t fnv1a(const void* address, std::size_t bytes, std::uint64_t seed = 1469598103934665603ULL) {
  const auto* data = static_cast<const unsigned char*>(address);
  std::uint64_t h = seed;
  for (std::size_t i = 0; i < bytes; ++i) { h ^= static_cast<std::uint64_t>(data[i]); h *= 1099511628211ULL; }
  return h;
}

struct FrozenTreeSnapshot {
  int node_count = 0;
  std::uint64_t hash = 0;
  void capture(const Runtime& runtime) {
    if (!runtime.max_node_num || !runtime.node_list || !runtime.id_list || !runtime.empty_list || !max_dis_d) fail("tree not initialized");
    node_count = runtime.max_node_num[0];
    if (node_count <= 0) fail("invalid tree node count");
    std::vector<TN> nodes(static_cast<std::size_t>(node_count));
    std::vector<int> empty(static_cast<std::size_t>(node_count));
    std::vector<float> maxd(static_cast<std::size_t>(node_count));
    std::vector<int> ids(kBaseN);
    C2_CUDA(cudaMemcpy(nodes.data(), runtime.node_list, nodes.size() * sizeof(TN), cudaMemcpyDeviceToHost));
    C2_CUDA(cudaMemcpy(empty.data(), runtime.empty_list, empty.size() * sizeof(int), cudaMemcpyDeviceToHost));
    C2_CUDA(cudaMemcpy(maxd.data(), max_dis_d, maxd.size() * sizeof(float), cudaMemcpyDeviceToHost));
    C2_CUDA(cudaMemcpy(ids.data(), runtime.id_list, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
    std::uint64_t h = fnv1a(&node_count, sizeof(node_count));
    h = fnv1a(nodes.data(), nodes.size() * sizeof(TN), h);
    h = fnv1a(empty.data(), empty.size() * sizeof(int), h);
    h = fnv1a(maxd.data(), maxd.size() * sizeof(float), h);
    h = fnv1a(ids.data(), ids.size() * sizeof(int), h);
    hash = h;
  }
  void assert_unchanged(const Runtime& runtime) const {
    FrozenTreeSnapshot other; other.capture(runtime);
    if (other.node_count != node_count || other.hash != hash) fail("frozen static tree mutated during C2 query");
  }
};

struct Timing {
  double wall_ms = 0.0;
  double gpu_ms = 0.0;
};

Timing elapsed_cuda_event(cudaEvent_t start, cudaEvent_t stop, std::chrono::steady_clock::time_point wall_start) {
  C2_CUDA(cudaEventRecord(stop));
  C2_CUDA(cudaEventSynchronize(stop));
  float gpu = 0.0F;
  C2_CUDA(cudaEventElapsedTime(&gpu, start, stop));
  Timing out;
  out.gpu_ms = gpu;
  out.wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall_start).count();
  return out;
}

struct BuildTiming {
  Timing base_h2d;
  Timing tree_build;
  double input_load_wall_ms = 0.0;
  double snapshot_wall_ms = 0.0;
};

Runtime build_tree(const Fvecs& base, const MetricPlan& metric, BuildTiming* timing) {
  Runtime runtime;
  C2_CUDA(cudaMallocManaged(&runtime.data_info, 3 * sizeof(int)));
  runtime.data_info[0] = kDimension;
  runtime.data_info[1] = kBaseN;
  runtime.data_info[2] = 2;
  C2_CUDA(cudaMalloc(&runtime.data_d, base.values.size() * sizeof(float)));
  cudaEvent_t begin = nullptr, end = nullptr;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  auto h2d_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  C2_CUDA(cudaMemcpyAsync(runtime.data_d, base.values.data(), base.values.size() * sizeof(float), cudaMemcpyHostToDevice));
  timing->base_h2d = elapsed_cuda_event(begin, end, h2d_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end)); begin = end = nullptr;
  DIS_CODE = 100;
  INFI_DIS = metric.infi_dis;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  auto tree_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  indexConstru(runtime.data_d, nullptr, nullptr, runtime.data_info, runtime.id_list, runtime.node_list,
               runtime.max_node_num, runtime.tree_height, runtime.empty_list);
  C2_CUDA(cudaDeviceSynchronize());
  C2_CUDA(cudaGetLastError());
  timing->tree_build = elapsed_cuda_event(begin, end, tree_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end));
  // The archived constructor uses these global temporary pointers internally.
  // They have already been freed in this code path; null them to avoid a false
  // double-free in isolated cleanup.  Tree outputs remain owned by Runtime.
  dis_list = nullptr; split_list = nullptr; split_num = nullptr; pid_list = nullptr;
  if (!runtime.id_list || !runtime.node_list || !runtime.max_node_num || !runtime.empty_list || runtime.tree_height <= 0) {
    fail("GTS tree build did not produce a valid runtime");
  }
  return runtime;
}

bool close_distance(float observed, float reference) {
  const float ulp = std::fabs(std::nextafterf(reference, std::numeric_limits<float>::infinity()) - reference);
  const float tolerance = std::max(kDistanceAbsFloor, static_cast<float>(kDistanceMaxUlps) * ulp);
  return std::fabs(observed - reference) <= tolerance;
}

float raw_l2(const float* a, const float* b) {
  float sum = 0.0F;
  for (int d = 0; d < kDimension; ++d) { const float delta = a[d] - b[d]; sum += delta * delta; }
  return std::sqrt(sum);
}

struct ReferenceCounters {
  long long correct = 0;
  long long total = 0;
  int invalid_output_ids = 0;
  int duplicate_output_ids = 0;
  int observed_distance_mismatches = 0;
  int reference_inconsistencies = 0;
  int boundary_ties = 0;
  std::string first_error;
  bool valid() const {
    return invalid_output_ids == 0 && duplicate_output_ids == 0 && observed_distance_mismatches == 0 &&
           reference_inconsistencies == 0 && boundary_ties == 0;
  }
};

void note_error(ReferenceCounters* c, const std::string& text) {
  if (c->first_error.empty()) c->first_error = text;
}

ReferenceCounters evaluate_results(const Fvecs& base, const Fvecs& all_queries, const std::vector<int>& gt,
                                   const std::vector<int>& selected_ids, const int* observed_ids,
                                   const float* observed_distances) {
  ReferenceCounters result;
  result.total = static_cast<long long>(selected_ids.size()) * kK;
  for (std::size_t local = 0; local < selected_ids.size(); ++local) {
    const int qid = selected_ids[local];
    const float* query = all_queries.values.data() + static_cast<std::size_t>(qid) * kDimension;
    const int* gt_row = gt.data() + static_cast<std::size_t>(qid) * kGtWidth;
    std::unordered_set<int> gt_topk;
    std::vector<std::pair<float, int>> top100;
    top100.reserve(kGtWidth);
    for (int j = 0; j < kGtWidth; ++j) {
      const int id = gt_row[j];
      const float d = raw_l2(base.values.data() + static_cast<std::size_t>(id) * kDimension, query);
      top100.emplace_back(d, id);
      if (j < kK) gt_topk.insert(id);
    }
    std::sort(top100.begin(), top100.end(), [](const auto& left, const auto& right) {
      return left.first != right.first ? left.first < right.first : left.second < right.second;
    });
    std::unordered_set<int> recomputed_topk;
    for (int j = 0; j < kK; ++j) recomputed_topk.insert(top100[j].second);
    if (recomputed_topk != gt_topk) {
      ++result.reference_inconsistencies;
      note_error(&result, "GT top-10 set disagrees with raw-distance sort within stored top-100 at q=" + std::to_string(qid));
    }
    if (top100[kK - 1].first == top100[kK].first) {
      ++result.boundary_ties;
      note_error(&result, "public GT has a top-k boundary tie at q=" + std::to_string(qid));
    }
    std::unordered_set<int> seen;
    for (int rank = 0; rank < kK; ++rank) {
      const std::size_t offset = local * kK + rank;
      const int id = observed_ids[offset];
      const float reported = observed_distances[offset];
      if (id < 0 || id >= kBaseN) {
        ++result.invalid_output_ids;
        note_error(&result, "GTS emitted invalid ID at q=" + std::to_string(qid));
        continue;
      }
      if (!seen.insert(id).second) {
        ++result.duplicate_output_ids;
        note_error(&result, "GTS emitted duplicate top-k ID at q=" + std::to_string(qid));
      }
      const float exact_for_observed = raw_l2(base.values.data() + static_cast<std::size_t>(id) * kDimension, query);
      if (!std::isfinite(reported) || !close_distance(reported, exact_for_observed)) {
        ++result.observed_distance_mismatches;
        note_error(&result, "reported distance does not match raw L2 for emitted ID at q=" + std::to_string(qid));
      }
      if (gt_topk.count(id) != 0 && seen.count(id) == 1) ++result.correct;
    }
  }
  return result;
}

struct QueryRun {
  Timing query;
  double d2h_ids_wall_ms = 0.0;
  double d2h_distances_wall_ms = 0.0;
  ReferenceCounters reference;
};

void upload_gamma(const std::array<float, RP_MAX_LEVELS>& gamma) {
  float alpha[RP_MAX_LEVELS]{};
  float beta[RP_MAX_LEVELS]{};
  bool active = false;
  for (int level = 0; level < RP_MAX_LEVELS; ++level) {
    if (!std::isfinite(gamma[level]) || gamma[level] < 1.0F) fail("invalid gamma value");
    if (gamma[level] > 1.0F) active = true;
  }
  float gamma_copy[RP_MAX_LEVELS];
  for (int level = 0; level < RP_MAX_LEVELS; ++level) gamma_copy[level] = gamma[level];
  upload_rp_constants(alpha, beta, gamma_copy, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, active ? 1 : 0);
}

QueryRun execute_once(const Runtime& runtime, float* query_d, int query_count, const Fvecs& base,
                      const Fvecs& all_queries, const std::vector<int>& gt,
                      const std::vector<int>& selected_ids, int* result_ids_d,
                      int* result_ids_h, float* result_distances_h, bool collect_results) {
  // GTS keeps update_disk as a global traversal state.  C2 static trials must
  // begin from exactly the same baseline state every time.
  update_disk = false;
  cudaEvent_t begin = nullptr, end = nullptr;
  C2_CUDA(cudaEventCreate(&begin)); C2_CUDA(cudaEventCreate(&end));
  auto wall_start = std::chrono::steady_clock::now();
  C2_CUDA(cudaEventRecord(begin));
  searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list, runtime.max_node_num,
                    query_d, result_ids_d, query_count, kK, runtime.tree_height,
                    runtime.data_info, runtime.empty_list, nullptr, nullptr);
  QueryRun run;
  run.query = elapsed_cuda_event(begin, end, wall_start);
  C2_CUDA(cudaEventDestroy(begin)); C2_CUDA(cudaEventDestroy(end));
  if (res_dis == nullptr) fail("legacy GTS query did not publish res_dis");
  if (collect_results) {
    auto copy_ids_start = std::chrono::steady_clock::now();
    C2_CUDA(cudaMemcpy(result_ids_h, result_ids_d, static_cast<std::size_t>(query_count) * kK * sizeof(int), cudaMemcpyDeviceToHost));
    run.d2h_ids_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_ids_start).count();
    auto copy_dist_start = std::chrono::steady_clock::now();
    // res_dis is legacy managed memory; cudaMemcpyDefault makes its migration
    // explicit and measures it separately from the static API timing.
    C2_CUDA(cudaMemcpy(result_distances_h, res_dis, static_cast<std::size_t>(query_count) * kK * sizeof(float), cudaMemcpyDefault));
    run.d2h_distances_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copy_dist_start).count();
    run.reference = evaluate_results(base, all_queries, gt, selected_ids, result_ids_h, result_distances_h);
  }
  C2_CUDA(cudaFree(res_dis));
  res_dis = nullptr;
  return run;
}

struct StageResult {
  int warmups = 0;
  int timed_reps = 0;
  double query_wall_ms_sum = 0.0;
  double query_gpu_ms_sum = 0.0;
  double d2h_ids_wall_ms_sum = 0.0;
  double d2h_distances_wall_ms_sum = 0.0;
  ReferenceCounters reference_sum;
  std::vector<ReferenceCounters> repetitions;
};

void add_counters(ReferenceCounters* target, const ReferenceCounters& source) {
  target->correct += source.correct; target->total += source.total;
  target->invalid_output_ids += source.invalid_output_ids;
  target->duplicate_output_ids += source.duplicate_output_ids;
  target->observed_distance_mismatches += source.observed_distance_mismatches;
  target->reference_inconsistencies += source.reference_inconsistencies;
  target->boundary_ties += source.boundary_ties;
  if (target->first_error.empty()) target->first_error = source.first_error;
}

StageResult run_stage(const Runtime& runtime, const FrozenTreeSnapshot& frozen, float* query_d, int query_count,
                      const Fvecs& base, const Fvecs& all_queries, const std::vector<int>& gt,
                      const std::vector<int>& selected_ids, int warmups, int timed_reps) {
  int* ids_d = nullptr; int* ids_h = nullptr; float* distances_h = nullptr;
  const std::size_t result_count = static_cast<std::size_t>(query_count) * kK;
  C2_CUDA(cudaMalloc(&ids_d, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&ids_h, result_count * sizeof(int)));
  C2_CUDA(cudaMallocHost(&distances_h, result_count * sizeof(float)));
  StageResult result; result.warmups = warmups; result.timed_reps = timed_reps;
  try {
    for (int i = 0; i < warmups; ++i) {
      (void)execute_once(runtime, query_d, query_count, base, all_queries, gt, selected_ids, ids_d, ids_h, distances_h, false);
      frozen.assert_unchanged(runtime);
    }
    for (int i = 0; i < timed_reps; ++i) {
      QueryRun one = execute_once(runtime, query_d, query_count, base, all_queries, gt, selected_ids, ids_d, ids_h, distances_h, true);
      frozen.assert_unchanged(runtime);
      result.query_wall_ms_sum += one.query.wall_ms;
      result.query_gpu_ms_sum += one.query.gpu_ms;
      result.d2h_ids_wall_ms_sum += one.d2h_ids_wall_ms;
      result.d2h_distances_wall_ms_sum += one.d2h_distances_wall_ms;
      add_counters(&result.reference_sum, one.reference);
      result.repetitions.push_back(one.reference);
    }
  } catch (...) {
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    cudaFree(ids_d); cudaFreeHost(ids_h); cudaFreeHost(distances_h); throw;
  }
  C2_CUDA(cudaFree(ids_d)); C2_CUDA(cudaFreeHost(ids_h)); C2_CUDA(cudaFreeHost(distances_h));
  return result;
}

bool stage_gate(const StageResult& stage, long long baseline_correct) {
  // A repeated timing stage passes only when every repetition individually meets
  // the frozen baseline and no output/reference ambiguity occurred.
  if (stage.repetitions.empty()) return false;
  for (const auto& counters : stage.repetitions) {
    if (!counters.valid() || counters.correct < baseline_correct) return false;
  }
  return true;
}

std::string json_escape(const std::string& raw) {
  std::ostringstream out;
  for (unsigned char c : raw) {
    if (c == '"' || c == '\\') out << '\\' << static_cast<char>(c);
    else if (c == '\n') out << "\\n";
    else if (c < 0x20) out << ' ';
    else out << static_cast<char>(c);
  }
  return out.str();
}

void write_gamma_json(std::ostream& out, const std::array<float, RP_MAX_LEVELS>& gamma) {
  out << '[';
  for (int l = 0; l < RP_MAX_LEVELS; ++l) { if (l) out << ','; out << std::setprecision(8) << gamma[l]; }
  out << ']';
}

void write_reference_json(std::ostream& out, const ReferenceCounters& c) {
  out << "{\"correct\":" << c.correct << ",\"total\":" << c.total
      << ",\"recall_at_10\":" << std::setprecision(10)
      << (c.total ? static_cast<double>(c.correct) / static_cast<double>(c.total) : 0.0)
      << ",\"invalid_output_ids\":" << c.invalid_output_ids
      << ",\"duplicate_output_ids\":" << c.duplicate_output_ids
      << ",\"observed_distance_mismatches\":" << c.observed_distance_mismatches
      << ",\"reference_inconsistencies\":" << c.reference_inconsistencies
      << ",\"boundary_ties\":" << c.boundary_ties
      << ",\"first_error\":\"" << json_escape(c.first_error) << "\"}";
}

void write_stage_json(std::ostream& out, const StageResult& stage, int query_count) {
  out << "{\"warmup_reps\":" << stage.warmups << ",\"timed_reps\":" << stage.timed_reps
      << ",\"query_count\":" << query_count
      << ",\"static_gts_api_wall_ms_sum\":" << stage.query_wall_ms_sum
      << ",\"static_gts_api_gpu_ms_sum\":" << stage.query_gpu_ms_sum
      << ",\"static_gts_api_wall_ms_per_query\":" << (stage.query_wall_ms_sum / (stage.timed_reps * query_count))
      << ",\"static_gts_api_gpu_ms_per_query\":" << (stage.query_gpu_ms_sum / (stage.timed_reps * query_count))
      << ",\"d2h_ids_wall_ms_sum\":" << stage.d2h_ids_wall_ms_sum
      << ",\"d2h_distances_wall_ms_sum\":" << stage.d2h_distances_wall_ms_sum
      << ",\"reference_sum\":";
  write_reference_json(out, stage.reference_sum);
  out << ",\"per_repetition\":[";
  for (std::size_t i = 0; i < stage.repetitions.size(); ++i) { if (i) out << ','; write_reference_json(out, stage.repetitions[i]); }
  out << "]}";
}

struct CalibrationRecord {
  int level = -1;
  int iteration = -1;
  float candidate = 1.0F;
  long long correct = 0;
  bool reference_valid = false;
  bool accepted = false;
};

void write_summary(const std::string& path, const Args& args, const BuildTiming& build, const Runtime& runtime,
                   const MetricPlan& metric, const std::array<float, RP_MAX_LEVELS>& gamma,
                   const StageResult& result, long long baseline_target, bool stage_pass,
                   const std::vector<CalibrationRecord>& calibration, double calibration_wall_ms,
                   const std::string& status, const std::string& gpu_name) {
  std::ofstream out(path);
  if (!out) fail("cannot write summary");
  out << std::setprecision(10);
  out << "{\n"
      << "  \"schema\": \"submitted-vldb-c2-perlevel-run-v1\",\n"
      << "  \"status\": \"" << status << "\",\n"
      << "  \"scope\": \"submitted-version C2 per-level calibrated pruning on frozen static SIFT-1M; empirical recall evaluation only\",\n"
      << "  \"mode\": \"" << args.mode << "\", \"stage\": \"" << args.stage << "\",\n"
      << "  \"archived_incremental_updater_used\": false,\n"
      << "  \"dynamic_exact_smoke_used\": false,\n"
      << "  \"workspace_cap_bytes\": " << kWorkspaceCapBytes << ",\n"
      << "  \"tree\": {\"base_n\": " << kBaseN << ", \"dimension\": " << kDimension
      << ", \"height\": " << runtime.tree_height << ", \"metric\": \"L2\", \"infi_dis\": " << metric.infi_dis
      << ", \"bbox_l2_upper_bound\": " << static_cast<double>(metric.bbox_l2) << "},\n"
      << "  \"gamma_vector\": ";
  write_gamma_json(out, gamma);
  out << ",\n  \"c2_algorithm1\": {\"freeze_level\": " << kFreezeLevel
      << ", \"binary_iterations\": " << kBinaryIterations << ", \"candidate_lo\": " << kCandidateLo
      << ", \"candidate_hi\": " << kCandidateHi << ", \"mu\": " << kMu
      << ", \"order\": \"h-1 down to 3\"},\n"
      << "  \"timing\": {\"base_h2d_wall_ms\": " << build.base_h2d.wall_ms
      << ", \"base_h2d_gpu_ms\": " << build.base_h2d.gpu_ms
      << ", \"tree_build_wall_ms\": " << build.tree_build.wall_ms
      << ", \"tree_build_gpu_ms\": " << build.tree_build.gpu_ms
      << ", \"tree_snapshot_wall_ms\": " << build.snapshot_wall_ms
      << ", \"calibration_wall_ms\": " << calibration_wall_ms << ", \"stage\": ";
  write_stage_json(out, result, static_cast<int>(result.reference_sum.total / std::max(1, result.timed_reps * kK)));
  out << "},\n  \"baseline_target_correct\": " << baseline_target
      << ", \"stage_gate_pass\": " << (stage_pass ? "true" : "false") << ",\n"
      << "  \"calibration_trace\": [";
  for (std::size_t i = 0; i < calibration.size(); ++i) {
    const auto& r = calibration[i]; if (i) out << ',';
    out << "{\"level\":" << r.level << ",\"iteration\":" << r.iteration << ",\"candidate\":" << r.candidate
        << ",\"correct\":" << r.correct << ",\"reference_valid\":" << (r.reference_valid ? "true" : "false")
        << ",\"accepted\":" << (r.accepted ? "true" : "false") << '}';
  }
  out << "],\n  \"gpu\": {\"name\": \"" << json_escape(gpu_name) << "\"},\n"
      << "  \"not_established\": [\"universal no-false-negative safety for gamma>1\", \"dynamic update/C3 behavior\", \"drift/recalibration deployment\", \"submitted-paper performance claim\"]\n"
      << "}\n";
  if (!out) fail("failed writing summary");
}

int main(int argc, char** argv) {
  Runtime runtime;
  float* selected_queries_h = nullptr;
  float* selected_queries_d = nullptr;
  try {
    const Args args = parse_args(argc, argv);
    // Require caller-created empty output directory, preventing accidental
    // overwrites of a frozen stage result.
    { std::ifstream probe(args.output_dir + "/summary.json"); if (probe.good()) fail("summary already exists in --out"); }
    const auto load_start = std::chrono::steady_clock::now();
    const Fvecs base = read_fvecs(args.base_fvecs, kBaseN, kDimension);
    const Fvecs all_queries = read_fvecs(args.query_fvecs, kQueryN, kDimension);
    const std::vector<int> gt = read_ivecs(args.groundtruth_ivecs, kQueryN, kGtWidth);
    const std::vector<int> selected_ids = read_query_ids(args.query_ids);
    const MetricPlan metric = metric_plan(base, all_queries, selected_ids);
    BuildTiming build;
    build.input_load_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - load_start).count();
    runtime = build_tree(base, metric, &build);
    if (runtime.tree_height < 4 || runtime.tree_height > RP_MAX_LEVELS) {
      fail("runtime tree height outside frozen per-level C2 protocol range");
    }
    const auto snapshot_start = std::chrono::steady_clock::now();
    FrozenTreeSnapshot frozen; frozen.capture(runtime);
    build.snapshot_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - snapshot_start).count();

    const int selected_count = static_cast<int>(selected_ids.size());
    C2_CUDA(cudaMallocHost(&selected_queries_h, static_cast<std::size_t>(selected_count) * kDimension * sizeof(float)));
    for (int local = 0; local < selected_count; ++local) {
      const float* source = all_queries.values.data() + static_cast<std::size_t>(selected_ids[local]) * kDimension;
      std::memcpy(selected_queries_h + static_cast<std::size_t>(local) * kDimension, source, kDimension * sizeof(float));
    }
    C2_CUDA(cudaMalloc(&selected_queries_d, static_cast<std::size_t>(selected_count) * kDimension * sizeof(float)));
    cudaEvent_t h2d_begin = nullptr, h2d_end = nullptr;
    C2_CUDA(cudaEventCreate(&h2d_begin)); C2_CUDA(cudaEventCreate(&h2d_end));
    const auto query_h2d_start = std::chrono::steady_clock::now();
    C2_CUDA(cudaEventRecord(h2d_begin));
    C2_CUDA(cudaMemcpyAsync(selected_queries_d, selected_queries_h,
                             static_cast<std::size_t>(selected_count) * kDimension * sizeof(float), cudaMemcpyHostToDevice));
    const Timing query_h2d = elapsed_cuda_event(h2d_begin, h2d_end, query_h2d_start);
    C2_CUDA(cudaEventDestroy(h2d_begin)); C2_CUDA(cudaEventDestroy(h2d_end));

    cudaDeviceProp prop{}; int device = 0; C2_CUDA(cudaGetDevice(&device)); C2_CUDA(cudaGetDeviceProperties(&prop, device));
    std::array<float, RP_MAX_LEVELS> gamma{}; gamma.fill(1.0F);
    StageResult stage_result;
    long long baseline_target = args.baseline_target_correct;
    std::vector<CalibrationRecord> calibration_trace;
    double calibration_wall_ms = 0.0;
    bool passed = false;

    if (args.mode == "calibrate") {
      const auto calibration_start = std::chrono::steady_clock::now();
      upload_gamma(gamma);
      const StageResult baseline = run_stage(runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
                                             selected_ids, args.warmup_reps, 1);
      if (baseline.repetitions.size() != 1 || !baseline.repetitions[0].valid()) {
        fail("calibration baseline reference check failed");
      }
      baseline_target = baseline.repetitions[0].correct;
      // Algorithm 1: deepest to shallowest; fixed previously calibrated deeper
      // scales stay active while the current level is searched.
      for (int level = runtime.tree_height - 1; level >= kFreezeLevel + 1; --level) {
        float lo = kCandidateLo, hi = kCandidateHi, best = 1.0F;
        for (int iter = 0; iter < kBinaryIterations; ++iter) {
          const float mid = (lo + hi) / 2.0F;
          gamma[level] = mid;
          upload_gamma(gamma);
          const StageResult candidate = run_stage(runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
                                                  selected_ids, 0, 1);
          const ReferenceCounters& c = candidate.repetitions.at(0);
          const bool accepted = c.valid() && c.correct >= baseline_target;
          calibration_trace.push_back(CalibrationRecord{level, iter, mid, c.correct, c.valid(), accepted});
          if (accepted) { best = mid; lo = mid; } else { hi = mid; }
        }
        gamma[level] = 1.0F + kMu * (best - 1.0F);
        upload_gamma(gamma);
        // Conservative post-margin confirmation (pre-registered extra guard).
        const StageResult confirm = run_stage(runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
                                              selected_ids, 0, 1);
        const ReferenceCounters& c = confirm.repetitions.at(0);
        const bool confirmed = c.valid() && c.correct >= baseline_target;
        calibration_trace.push_back(CalibrationRecord{level, kBinaryIterations, gamma[level], c.correct, c.valid(), confirmed});
        if (!confirmed) { gamma[level] = 1.0F; upload_gamma(gamma); }
      }
      calibration_wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - calibration_start).count();
      // A final measurement stage is not used for selection; it records the
      // immutable calibrated vector once after all per-level choices are frozen.
      stage_result = run_stage(runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
                               selected_ids, 0, 1);
      passed = stage_gate(stage_result, baseline_target);
      write_gamma_vector(args.output_dir + "/final_gamma_vector.txt", gamma);
    } else {
      gamma = read_gamma_vector(args.gamma_vector_file);
      for (int level = 0; level <= kFreezeLevel; ++level) {
        if (gamma[level] != 1.0F) fail("gamma vector violates frozen shallow levels");
      }
      upload_gamma(gamma);
      stage_result = run_stage(runtime, frozen, selected_queries_d, selected_count, base, all_queries, gt,
                               selected_ids, args.warmup_reps, args.timed_reps);
      passed = stage_gate(stage_result, baseline_target);
    }

    // Query H2D is independent of GTS static API timing; attach it to build so
    // summary consumers cannot accidentally fold it into per-query latency.
    build.input_load_wall_ms += 0.0;  // retained separately in future schema revisions
    const std::string status = passed ? (args.mode == "calibrate" ? "PASS_C2_CALIBRATION" : "PASS_C2_HELDOUT_STAGE")
                                      : (args.mode == "calibrate" ? "FAIL_C2_CALIBRATION" : "FAIL_C2_HELDOUT_STAGE");
    // Store query H2D in an adjacent compact file; summary also gets it patched below.
    write_summary(args.output_dir + "/summary.json", args, build, runtime, metric, gamma, stage_result,
                  baseline_target, passed, calibration_trace, calibration_wall_ms, status, prop.name);
    {
      std::ofstream qh2d(args.output_dir + "/query_h2d.json");
      qh2d << std::setprecision(10) << "{\"query_count\":" << selected_count << ",\"query_h2d_wall_ms\":"
           << query_h2d.wall_ms << ",\"query_h2d_gpu_ms\":" << query_h2d.gpu_ms
           << ",\"note\":\"separate from static GTS query API timing\"}\n";
    }
    C2_CUDA(cudaFree(selected_queries_d)); selected_queries_d = nullptr;
    C2_CUDA(cudaFreeHost(selected_queries_h)); selected_queries_h = nullptr;
    release_tree(runtime);
    return passed ? 0 : 3;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    if (selected_queries_d) cudaFree(selected_queries_d);
    if (selected_queries_h) cudaFreeHost(selected_queries_h);
    release_tree(runtime);
    return 2;
  }
}
