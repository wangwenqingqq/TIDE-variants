// C3 native certified direct-insert correctness probe.
//
// Scope is deliberately narrow: a frozen GTS base tree, insert-only candidates
// that pass a strict root-to-leaf interval certificate, real padded id_list
// append + TN.size mutation, and immediate visibility through GTS's static
// vector top-K API.  It does NOT include the archived incremental updater,
// buffer merge, delete, range, rebuild, concurrency, or performance logic.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tree.cuh"
#include "search_v2.cuh"

namespace {

constexpr char kTraceMagic[8] = {'E', '1', 'G', 'T', 'R', 'C', '0', '1'};
constexpr std::uint32_t kTraceVersion = 1;
constexpr float kStrictEpsilon = 1.0e-4F;
constexpr float kDistanceAbsoluteFloor = 2.0e-3F;
constexpr int kDistanceMaxFloatUlps = 2;
constexpr int kMaxAccepted = 16;

#pragma pack(push, 1)
struct TraceHeader {
  char magic[8];
  std::uint32_t version;
  std::uint32_t dimension;
  std::uint32_t base_n;
  std::uint32_t reservoir_n;
  std::uint32_t pool_n;
  std::uint32_t query_n;
  std::uint32_t k;
  float radius;
  std::uint64_t event_count;
};
#pragma pack(pop)
static_assert(sizeof(TraceHeader) == 48, "unexpected E1 trace header ABI");

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("C3-native-probe: " + message);
}

void cuda_or_fail(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << "CUDA " << expression << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(status);
  fail(out.str());
}
#define C3_CUDA(expr) cuda_or_fail((expr), #expr, __FILE__, __LINE__)

struct Args {
  std::string bundle;
  std::string output;
  std::string summary;
  std::string initial_geometry;
  std::string final_geometry;
  int max_accepted = kMaxAccepted;
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto value = [&]() -> std::string {
      if (++i >= argc) fail("missing value after " + key);
      return argv[i];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--out") args.output = value();
    else if (key == "--summary") args.summary = value();
    else if (key == "--initial-geometry") args.initial_geometry = value();
    else if (key == "--final-geometry") args.final_geometry = value();
    else if (key == "--max-accepted") args.max_accepted = std::stoi(value());
    else if (key == "--help") {
      std::cout << "usage: GTS_c3_native_direct_insert_probe --bundle BUNDLE --out ENGINE.jsonl "
                   "--summary SUMMARY.json --initial-geometry INITIAL.json --final-geometry FINAL.json "
                   "[--max-accepted 16]\n";
      std::exit(0);
    } else {
      fail("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.output.empty() || args.summary.empty() ||
      args.initial_geometry.empty() || args.final_geometry.empty()) {
    fail("--bundle, --out, --summary, --initial-geometry, and --final-geometry are required");
  }
  if (args.max_accepted <= 0 || args.max_accepted > kMaxAccepted) {
    fail("--max-accepted must be in [1,16] to match the pre-registered protocol");
  }
  return args;
}

template <typename T>
std::vector<T> read_exact_binary(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open " + path);
  std::vector<T> values(count);
  input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    fail("unexpected byte length for " + path);
  }
  char extra = 0;
  if (input.read(&extra, 1)) fail("extra bytes in " + path);
  return values;
}

TraceHeader read_header(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) fail("cannot open trace " + path);
  TraceHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) fail("truncated trace header");
  if (std::memcmp(header.magic, kTraceMagic, sizeof(kTraceMagic)) != 0 || header.version != kTraceVersion) {
    fail("unsupported trace magic/version");
  }
  if (header.dimension == 0 || header.base_n == 0 || header.pool_n < header.base_n ||
      header.reservoir_n != header.pool_n - header.base_n || header.k == 0 ||
      header.k > header.base_n) {
    fail("invalid trace header");
  }
  return header;
}

struct MetricEncodingPlan {
  std::uint64_t bbox_squared_upper_bound = 0;
  long double bbox_l2_upper_bound = 0.0L;
  int infi_dis = 0;
};

MetricEncodingPlan make_metric_encoding_plan(const std::vector<std::int16_t>& pool,
                                              const std::vector<std::int16_t>& queries,
                                              int dimension) {
  if (dimension <= 0 || pool.empty() || queries.empty()) fail("malformed metric inputs");
  std::vector<int> lower(dimension, std::numeric_limits<int>::max());
  std::vector<int> upper(dimension, std::numeric_limits<int>::min());
  auto include = [&](const std::vector<std::int16_t>& values) {
    for (std::size_t i = 0; i < values.size(); ++i) {
      const int axis = static_cast<int>(i % static_cast<std::size_t>(dimension));
      lower[axis] = std::min(lower[axis], static_cast<int>(values[i]));
      upper[axis] = std::max(upper[axis], static_cast<int>(values[i]));
    }
  };
  include(pool); include(queries);
  std::uint64_t squared = 0;
  for (int d = 0; d < dimension; ++d) {
    const std::int64_t span = static_cast<std::int64_t>(upper[d]) - lower[d];
    const std::uint64_t term = static_cast<std::uint64_t>(span * span);
    if (squared > std::numeric_limits<std::uint64_t>::max() - term) fail("metric bound overflow");
    squared += term;
  }
  const long double bound = std::sqrt(static_cast<long double>(squared));
  const long double requested = std::ceil(bound) + 1024.0L;
  if (!std::isfinite(requested) || requested > (1 << 24)) fail("invalid INFI_DIS bound");
  return MetricEncodingPlan{squared, bound, std::max(10000, static_cast<int>(requested))};
}

struct Runtime {
  int* data_info = nullptr;
  short* data_d = nullptr;
  int* id_list = nullptr;
  TN* node_list = nullptr;
  int* max_node_num = nullptr;
  int* empty_list = nullptr;
  int tree_height = 0;
  int base_count = 0;

  bool ready() const {
    return data_info != nullptr && data_d != nullptr && id_list != nullptr &&
           node_list != nullptr && max_node_num != nullptr && empty_list != nullptr &&
           tree_height > 1 && base_count > 0;
  }
};

void release_runtime(Runtime& runtime) {
  if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
  if (runtime.id_list) { cudaFree(runtime.id_list); runtime.id_list = nullptr; }
  if (runtime.node_list) { cudaFree(runtime.node_list); runtime.node_list = nullptr; }
  if (runtime.max_node_num) { cudaFree(runtime.max_node_num); runtime.max_node_num = nullptr; }
  if (runtime.empty_list) { cudaFree(runtime.empty_list); runtime.empty_list = nullptr; }
  if (runtime.data_d) { cudaFree(runtime.data_d); runtime.data_d = nullptr; }
  if (runtime.data_info) { cudaFree(runtime.data_info); runtime.data_info = nullptr; }
  if (max_dis_d) { cudaFree(max_dis_d); max_dis_d = nullptr; }
  // indexConstru owns/frees these temporaries; callers null them after construction.
  dis_list = nullptr; split_list = nullptr; split_num = nullptr; pid_list = nullptr;
}

std::uint32_t float_bits(float value) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

struct Snapshot {
  int tree_height = 0;
  int tree_order = 0;
  int max_size = 0;
  int leaf_pad_slots = 0;
  std::vector<TN> nodes;
  std::vector<int> empty;
  std::vector<float> max_distance;
  std::vector<std::vector<int>> leaf_ids;  // indexed by node ID; non-leaves empty.
};

Snapshot capture_snapshot(const Runtime& runtime) {
  if (!runtime.ready()) fail("cannot snapshot unready runtime");
  const int count = runtime.max_node_num[0];
  if (count <= 0) fail("invalid max node count");
  Snapshot snapshot;
  snapshot.tree_height = runtime.tree_height;
  snapshot.tree_order = TREE_ORDER;
  snapshot.max_size = MAX_SIZE;
  snapshot.leaf_pad_slots = LEAF_PAD_SLOTS;
  snapshot.nodes.resize(count);
  snapshot.empty.resize(count);
  snapshot.max_distance.resize(count);
  snapshot.leaf_ids.resize(count);
  C3_CUDA(cudaMemcpy(snapshot.nodes.data(), runtime.node_list, count * sizeof(TN), cudaMemcpyDeviceToHost));
  C3_CUDA(cudaMemcpy(snapshot.empty.data(), runtime.empty_list, count * sizeof(int), cudaMemcpyDeviceToHost));
  C3_CUDA(cudaMemcpy(snapshot.max_distance.data(), max_dis_d, count * sizeof(float), cudaMemcpyDeviceToHost));
  for (int node = 0; node < count; ++node) {
    if (snapshot.empty[node] != 0 || snapshot.nodes[node].is_leaf != 1) continue;
    const TN& leaf = snapshot.nodes[node];
    if (leaf.size < 0 || leaf.lid < 0) fail("invalid leaf layout in snapshot");
    snapshot.leaf_ids[node].resize(static_cast<std::size_t>(leaf.size));
    if (!snapshot.leaf_ids[node].empty()) {
      C3_CUDA(cudaMemcpy(snapshot.leaf_ids[node].data(), runtime.id_list + leaf.lid,
                         snapshot.leaf_ids[node].size() * sizeof(int), cudaMemcpyDeviceToHost));
    }
  }
  return snapshot;
}

void write_snapshot_json(const std::string& path, const Snapshot& snapshot) {
  std::ofstream output(path);
  if (!output) fail("cannot write geometry snapshot " + path);
  output << "{\n  \"schema\":\"c3-native-gts-geometry-snapshot-v1\",\n"
         << "  \"tree_height\":" << snapshot.tree_height << ",\n"
         << "  \"tree_order\":" << snapshot.tree_order << ",\n"
         << "  \"max_size\":" << snapshot.max_size << ",\n"
         << "  \"leaf_pad_slots\":" << snapshot.leaf_pad_slots << ",\n"
         << "  \"nodes\":[\n";
  for (std::size_t i = 0; i < snapshot.nodes.size(); ++i) {
    const TN& node = snapshot.nodes[i];
    output << "    {\"node_id\":" << i << ",\"empty\":" << snapshot.empty[i]
           << ",\"pid\":" << node.pid << ",\"min_dis_bits\":" << float_bits(node.min_dis)
           << ",\"max_dis_bits\":" << float_bits(snapshot.max_distance[i])
           << ",\"size\":" << node.size << ",\"lid\":" << node.lid
           << ",\"is_leaf\":" << node.is_leaf;
    if (snapshot.empty[i] == 0 && node.is_leaf == 1) {
      output << ",\"logical_ids\":[";
      for (std::size_t j = 0; j < snapshot.leaf_ids[i].size(); ++j) {
        if (j) output << ',';
        output << snapshot.leaf_ids[i][j];
      }
      output << ']';
    }
    output << '}';
    if (i + 1 != snapshot.nodes.size()) output << ',';
    output << '\n';
  }
  output << "  ]\n}\n";
  if (!output) fail("write failed for geometry snapshot " + path);
}

void assert_native_layout(const Snapshot& initial, const Snapshot& current,
                          const std::map<int, std::vector<int>>& appended) {
  if (initial.tree_height != current.tree_height || initial.tree_order != current.tree_order ||
      initial.nodes.size() != current.nodes.size()) {
    fail("tree topology changed after native insert");
  }
  for (std::size_t i = 0; i < initial.nodes.size(); ++i) {
    const TN& before = initial.nodes[i];
    const TN& after = current.nodes[i];
    if (initial.empty[i] != current.empty[i] || before.pid != after.pid ||
        float_bits(before.min_dis) != float_bits(after.min_dis) || before.lid != after.lid ||
        before.is_leaf != after.is_leaf || float_bits(initial.max_distance[i]) != float_bits(current.max_distance[i])) {
      fail("frozen geometry changed at node " + std::to_string(i));
    }
    const auto it = appended.find(static_cast<int>(i));
    const int added = it == appended.end() ? 0 : static_cast<int>(it->second.size());
    if (before.is_leaf != 1 || initial.empty[i] != 0) {
      if (before.size != after.size) fail("non-leaf size changed at node " + std::to_string(i));
      continue;
    }
    if (after.size != before.size + added) {
      fail("leaf size inconsistent with native direct writes at node " + std::to_string(i));
    }
    std::vector<int> expected = initial.leaf_ids[i];
    if (it != appended.end()) expected.insert(expected.end(), it->second.begin(), it->second.end());
    if (current.leaf_ids[i] != expected) {
      fail("logical leaf id_list changed outside append-only native writes at node " + std::to_string(i));
    }
  }
}

float l2_routing_float(const std::vector<std::int16_t>& pool, int dimension, int left, int right) {
  float sum = 0.0F;
  const std::size_t lhs = static_cast<std::size_t>(left) * dimension;
  const std::size_t rhs = static_cast<std::size_t>(right) * dimension;
  for (int d = 0; d < dimension; ++d) {
    const float delta = static_cast<float>(pool[lhs + d]) - static_cast<float>(pool[rhs + d]);
    sum += delta * delta;
  }
  return std::sqrt(sum);
}

struct PathHop {
  int parent = -1;
  int pivot = -1;
  int child = -1;
  int matching_children = 0;
  float distance = 0.0F;
  float lower = 0.0F;
  float max_upper = 0.0F;
  float search_upper = 0.0F;
  bool has_next_sibling_upper = false;
};

struct Certificate {
  bool accepted = false;
  int leaf = -1;
  int pre_leaf_size = -1;
  std::string reason;
  std::vector<PathHop> path;
};

// search_v2::nodeProcessKnn does not read max_dis_d. For every node whose
// index is not the final child slot, its second lower-bound term is
// dis_q - node_list[nid + 1].min_dis.  This audit validates the hidden
// ordering contract instead of assuming max_dis_d is the search upper bound.
bool verify_search_upper_invariant(const Snapshot& frozen, std::string* detail) {
  for (int parent = 0; parent < static_cast<int>(frozen.nodes.size()); ++parent) {
    if (frozen.empty[parent] != 0) continue;
    for (int slot = 0; slot < frozen.tree_order; ++slot) {
      const int child = parent * frozen.tree_order + slot + 1;
      if (child < 0 || child >= static_cast<int>(frozen.nodes.size()) || frozen.empty[child] != 0) continue;
      const float lower = frozen.nodes[child].min_dis;
      const float max_upper = frozen.max_distance[child];
      if (!(lower <= max_upper)) {
        if (detail) *detail = "min_dis exceeds max_dis_d at child " + std::to_string(child);
        return false;
      }
      // Exact branch condition in nodeProcessKnn: nid % TREE_ORDER != 0.
      if (child % frozen.tree_order == 0) continue;
      const int next = child + 1;
      if (next < 0 || next >= static_cast<int>(frozen.nodes.size()) || frozen.empty[next] != 0 ||
          (next - 1) / frozen.tree_order != parent) {
        if (detail) *detail = "missing/non-sibling next child for node " + std::to_string(child);
        return false;
      }
      const float next_min = frozen.nodes[next].min_dis;
      if (!(max_upper <= next_min)) {
        if (detail) {
          *detail = "max_dis_d exceeds next_sibling.min_dis at child " + std::to_string(child) +
                    " (max=" + std::to_string(max_upper) + ", next_min=" + std::to_string(next_min) + ')';
        }
        return false;
      }
    }
  }
  return true;
}

Certificate certify_candidate(const Snapshot& frozen, const std::vector<std::int16_t>& pool,
                              int dimension, int stable_id,
                              const std::map<int, std::vector<int>>& appended,
                              bool search_upper_invariant_ok) {
  Certificate result;
  if (!search_upper_invariant_ok) {
    result.reason = "search_upper_invariant";
    return result;
  }
  if (frozen.tree_height <= 1 || frozen.tree_order <= 1) {
    result.reason = "missing_or_invalid_path";
    return result;
  }
  int parent = 0;
  for (int level = 0; level < frozen.tree_height - 1; ++level) {
    int pivot = -1;
    bool inconsistent_pivot = false;
    for (int slot = 0; slot < frozen.tree_order; ++slot) {
      const int child = parent * frozen.tree_order + slot + 1;
      if (child < 0 || child >= static_cast<int>(frozen.nodes.size()) || frozen.empty[child] != 0) continue;
      if (pivot < 0) pivot = frozen.nodes[child].pid;
      else if (frozen.nodes[child].pid != pivot) inconsistent_pivot = true;
    }
    if (pivot < 0 || inconsistent_pivot || pivot == stable_id) {
      result.reason = "missing_or_invalid_path";
      return result;
    }
    const float distance = l2_routing_float(pool, dimension, stable_id, pivot);
    int match = -1;
    int matches = 0;
    float matched_lower = 0.0F, matched_max_upper = 0.0F, matched_search_upper = 0.0F;
    bool matched_has_next = false;
    for (int slot = 0; slot < frozen.tree_order; ++slot) {
      const int child = parent * frozen.tree_order + slot + 1;
      if (child < 0 || child >= static_cast<int>(frozen.nodes.size()) || frozen.empty[child] != 0) continue;
      const float lower = frozen.nodes[child].min_dis;
      const float max_upper = frozen.max_distance[child];
      const bool has_next = child % frozen.tree_order != 0;
      float actual_search_upper = max_upper;
      if (has_next) {
        const int next = child + 1;
        if (next < 0 || next >= static_cast<int>(frozen.nodes.size()) || frozen.empty[next] != 0 ||
            (next - 1) / frozen.tree_order != parent) {
          result.reason = "search_upper_invariant";
          return result;
        }
        actual_search_upper = frozen.nodes[next].min_dis;
      }
      // Stronger than the actual static pruning branch: require strict interior
      // of both the stored max_dis interval and, when present, the next-sibling
      // min_dis search upper bound.
      const bool interior_max = distance > lower + kStrictEpsilon && distance < max_upper - kStrictEpsilon;
      const bool interior_search = !has_next || distance < actual_search_upper - kStrictEpsilon;
      if (interior_max && interior_search) {
        ++matches;
        match = child;
        matched_lower = lower;
        matched_max_upper = max_upper;
        matched_search_upper = actual_search_upper;
        matched_has_next = has_next;
      }
    }
    result.path.push_back(PathHop{parent, pivot, match, matches, distance, matched_lower,
                                  matched_max_upper, matched_search_upper, matched_has_next});
    if (matches != 1 || match < 0) {
      result.reason = "no_unique_strict_child";
      return result;
    }
    if (frozen.nodes[match].is_leaf == 1) {
      result.leaf = match;
      break;
    }
    parent = match;
  }
  if (result.leaf < 0 || frozen.empty[result.leaf] != 0 || frozen.nodes[result.leaf].is_leaf != 1) {
    result.reason = "not_leaf";
    return result;
  }
  const auto existing = appended.find(result.leaf);
  const int appended_here = existing == appended.end() ? 0 : static_cast<int>(existing->second.size());
  result.pre_leaf_size = frozen.nodes[result.leaf].size + appended_here;
  if (result.pre_leaf_size >= frozen.max_size) {
    result.reason = "static_scan_limit";
    return result;
  }
  if (appended_here >= frozen.leaf_pad_slots) {
    result.reason = "padded_leaf_capacity";
    return result;
  }
  result.accepted = true;
  result.reason = "accepted";
  return result;
}

void native_append(Runtime& runtime, int leaf_id, int expected_pre_size, int stable_id) {
  TN leaf{};
  C3_CUDA(cudaMemcpy(&leaf, runtime.node_list + leaf_id, sizeof(TN), cudaMemcpyDeviceToHost));
  if (leaf.is_leaf != 1 || leaf.size != expected_pre_size || leaf.lid < 0) {
    fail("native append precondition failed for leaf " + std::to_string(leaf_id));
  }
  if (leaf.size >= MAX_SIZE) fail("native append would exceed static leaf scan limit");
  const int slot = leaf.lid + leaf.size;
  C3_CUDA(cudaMemcpy(runtime.id_list + slot, &stable_id, sizeof(int), cudaMemcpyHostToDevice));
  ++leaf.size;
  C3_CUDA(cudaMemcpy(runtime.node_list + leaf_id, &leaf, sizeof(TN), cudaMemcpyHostToDevice));
  C3_CUDA(cudaDeviceSynchronize());
}

struct ExactResult {
  std::vector<int> ids;
  std::vector<std::int64_t> squared_distances;
  bool boundary_tie = false;
};

ExactResult exact_topk(const std::vector<std::int16_t>& pool, int dimension,
                       int query_id, const std::vector<std::uint8_t>& active, int k) {
  struct Pair { std::int64_t squared; int id; };
  std::vector<Pair> values;
  values.reserve(active.size());
  const std::size_t qoff = static_cast<std::size_t>(query_id) * dimension;
  for (int id = 0; id < static_cast<int>(active.size()); ++id) {
    if (active[id] == 0) continue;
    const std::size_t off = static_cast<std::size_t>(id) * dimension;
    std::int64_t sum = 0;
    for (int d = 0; d < dimension; ++d) {
      const std::int64_t delta = static_cast<std::int64_t>(pool[off + d]) - static_cast<std::int64_t>(pool[qoff + d]);
      sum += delta * delta;
    }
    values.push_back(Pair{sum, id});
  }
  if (values.size() < static_cast<std::size_t>(k)) fail("exact active set has fewer than k entries");
  std::sort(values.begin(), values.end(), [](const Pair& a, const Pair& b) {
    return a.squared != b.squared ? a.squared < b.squared : a.id < b.id;
  });
  ExactResult result;
  result.boundary_tie = values.size() > static_cast<std::size_t>(k) && values[k - 1].squared == values[k].squared;
  for (int i = 0; i < k; ++i) {
    result.ids.push_back(values[i].id);
    result.squared_distances.push_back(values[i].squared);
  }
  return result;
}

bool close_distance(float observed, float reference) {
  const float magnitude = std::max({1.0F, std::fabs(observed), std::fabs(reference)});
  const float ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
  return std::fabs(observed - reference) <= std::max(kDistanceAbsoluteFloor,
      static_cast<float>(kDistanceMaxFloatUlps) * ulp);
}

struct GtsResult {
  std::vector<int> ids;
  std::vector<float> distances;
};

GtsResult run_real_static_topk(Runtime& runtime, const std::vector<std::int16_t>& pool,
                               int dimension, int query_id, int k) {
  float* query = nullptr;
  int* result_ids = nullptr;
  try {
    C3_CUDA(cudaMallocManaged(&query, static_cast<std::size_t>(dimension) * sizeof(float)));
    const std::size_t offset = static_cast<std::size_t>(query_id) * dimension;
    for (int d = 0; d < dimension; ++d) query[d] = static_cast<float>(pool[offset + d]);
    C3_CUDA(cudaMallocManaged(&result_ids, static_cast<std::size_t>(k) * sizeof(int)));
    update_disk = false;
    searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list, runtime.max_node_num,
                      query, result_ids, 1, k, runtime.tree_height, runtime.data_info,
                      runtime.empty_list, nullptr, nullptr);
    C3_CUDA(cudaDeviceSynchronize());
    C3_CUDA(cudaGetLastError());
    GtsResult result;
    result.ids.assign(result_ids, result_ids + k);
    result.distances.assign(res_dis, res_dis + k);
    C3_CUDA(cudaFree(result_ids)); result_ids = nullptr;
    C3_CUDA(cudaFree(res_dis)); res_dis = nullptr;
    C3_CUDA(cudaFree(query)); query = nullptr;
    return result;
  } catch (...) {
    if (result_ids) cudaFree(result_ids);
    if (res_dis) { cudaFree(res_dis); res_dis = nullptr; }
    if (query) cudaFree(query);
    throw;
  }
}

void validate_real_query(const GtsResult& gts, const ExactResult& exact, int inserted_id) {
  if (exact.boundary_tie) fail("k-boundary tie blocks C3 visibility proof");
  if (gts.ids.size() != exact.ids.size() || gts.distances.size() != exact.ids.size()) {
    fail("static GTS top-k output length mismatch");
  }
  for (std::size_t begin = 0; begin < exact.ids.size();) {
    std::size_t end = begin + 1;
    while (end < exact.ids.size() && exact.squared_distances[end] == exact.squared_distances[begin]) ++end;
    std::vector<int> expected_ids(exact.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                  exact.ids.begin() + static_cast<std::ptrdiff_t>(end));
    std::vector<int> observed_ids(gts.ids.begin() + static_cast<std::ptrdiff_t>(begin),
                                  gts.ids.begin() + static_cast<std::ptrdiff_t>(end));
    std::sort(expected_ids.begin(), expected_ids.end());
    std::sort(observed_ids.begin(), observed_ids.end());
    if (expected_ids != observed_ids) fail("real static GTS top-k ID mismatch");
    for (std::size_t pos = begin; pos < end; ++pos) {
      const float reference = static_cast<float>(std::sqrt(static_cast<long double>(exact.squared_distances[pos])));
      if (!close_distance(gts.distances[pos], reference)) fail("real static GTS top-k distance mismatch");
    }
    begin = end;
  }
  auto found = std::find(gts.ids.begin(), gts.ids.end(), inserted_id);
  if (found == gts.ids.end()) fail("accepted native ID missing from real static GTS self-query");
  const std::size_t position = static_cast<std::size_t>(found - gts.ids.begin());
  if (gts.distances[position] != 0.0F) fail("accepted native ID is visible but nonzero at self-query");
}

std::string json_number(double value) {
  std::ostringstream out;
  out << std::setprecision(17) << value;
  return out.str();
}

void emit_int_array(std::ostream& out, const std::vector<int>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
void emit_i64_array(std::ostream& out, const std::vector<std::int64_t>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << values[i]; }
  out << ']';
}
void emit_float_array(std::ostream& out, const std::vector<float>& values) {
  out << '[';
  for (std::size_t i = 0; i < values.size(); ++i) { if (i) out << ','; out << json_number(values[i]); }
  out << ']';
}
void emit_path(std::ostream& out, const std::vector<PathHop>& path) {
  out << '[';
  for (std::size_t i = 0; i < path.size(); ++i) {
    if (i) out << ',';
    const PathHop& p = path[i];
    out << "{\"parent\":" << p.parent << ",\"pivot\":" << p.pivot << ",\"child\":" << p.child
        << ",\"matching_children\":" << p.matching_children << ",\"distance\":" << json_number(p.distance)
        << ",\"lower\":" << json_number(p.lower) << ",\"max_upper\":" << json_number(p.max_upper)
        << ",\"search_upper\":" << json_number(p.search_upper)
        << ",\"has_next_sibling_upper\":" << (p.has_next_sibling_upper ? "true" : "false") << '}';
  }
  out << ']';
}

void emit_candidate(std::ofstream& output, int stable_id, const Certificate& certificate,
                    int leaf_lid, const std::vector<int>* before, const std::vector<int>* after) {
  output << "{\"record\":\"candidate\",\"stable_id\":" << stable_id
         << ",\"outcome\":\"" << (certificate.accepted ? "accepted" : "rejected") << "\""
         << ",\"reason\":\"" << certificate.reason << "\""
         << ",\"leaf_id\":" << certificate.leaf << ",\"pre_leaf_size\":" << certificate.pre_leaf_size
         << ",\"path\":";
  emit_path(output, certificate.path);
  if (certificate.accepted && before != nullptr && after != nullptr) {
    output << ",\"leaf_lid\":" << leaf_lid
           << ",\"write_slot\":" << (leaf_lid + certificate.pre_leaf_size)
           << ",\"leaf_ids_before\":";
    emit_int_array(output, *before);
    output << ",\"leaf_ids_after\":";
    emit_int_array(output, *after);
  }
  output << "}\n";
}

void emit_self_query(std::ofstream& output, int stable_id, int active_count,
                     const GtsResult& gts, const ExactResult& exact) {
  output << "{\"record\":\"self_query\",\"stable_id\":" << stable_id
         << ",\"active_count\":" << active_count << ",\"gts_ids\":";
  emit_int_array(output, gts.ids);
  output << ",\"gts_distances\":";
  emit_float_array(output, gts.distances);
  output << ",\"runner_exact_ids\":";
  emit_int_array(output, exact.ids);
  output << ",\"runner_exact_squared_distances\":";
  emit_i64_array(output, exact.squared_distances);
  output << ",\"k_boundary_tie\":" << (exact.boundary_tie ? "true" : "false") << "}\n";
}

int active_count(const std::vector<std::uint8_t>& active) {
  return static_cast<int>(std::count(active.begin(), active.end(), static_cast<std::uint8_t>(1)));
}

void write_summary(const std::string& path, const std::string& status, const std::string& detail,
                   const TraceHeader* header, int candidates, int accepted, int rejected,
                   const MetricEncodingPlan* plan) {
  std::ofstream out(path);
  if (!out) return;
  out << "{\n  \"schema\":\"c3-native-certified-direct-insert-runner-v1\",\n"
      << "  \"status\":\"" << status << "\",\n"
      << "  \"scope\":\"native certified direct-insert correctness probe only; no performance or complete C3 claim\",\n"
      << "  \"gpu_execution\":true,\n"
      << "  \"archived_incremental_updater_used\":false,\n"
      << "  \"accepted\":" << accepted << ",\n  \"rejected\":" << rejected
      << ",\n  \"candidates_examined\":" << candidates;
  if (header != nullptr) out << ",\n  \"base_n\":" << header->base_n << ",\n  \"pool_n\":" << header->pool_n
                              << ",\n  \"dimension\":" << header->dimension << ",\n  \"k\":" << header->k;
  if (plan != nullptr) out << ",\n  \"metric_encoding\":{\"infi_dis\":" << plan->infi_dis
                            << ",\"bbox_squared_upper_bound\":" << plan->bbox_squared_upper_bound << "}";
  out << ",\n  \"detail\":\"" << detail << "\",\n"
      << "  \"limitations\":[\"insert-only direct tier\",\"rejected-buffer merge untested\",\"no range/delete/concurrency/rebuild\",\"not performance evidence\",\"CPU validator required for PASS\"]\n}\n";
}

}  // namespace

int main(int argc, char** argv) {
  Runtime runtime;
  Args args;
  TraceHeader header{};
  MetricEncodingPlan plan{};
  int candidates = 0, accepted = 0, rejected = 0;
  try {
    args = parse_args(argc, argv);
    header = read_header(args.bundle + "/trace.e1gtrc");
    const std::size_t pool_values = static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values = static_cast<std::size_t>(header.query_n) * header.dimension;
    const auto pool = read_exact_binary<std::int16_t>(args.bundle + "/pool.i16", pool_values);
    const auto queries = read_exact_binary<std::int16_t>(args.bundle + "/queries.i16", query_values);
    plan = make_metric_encoding_plan(pool, queries, static_cast<int>(header.dimension));
    DIS_CODE = 100;
    INFI_DIS = plan.infi_dis;

    // Build a base-only tree but allocate the entire immutable pool so accepted
    // reservoir stable IDs have real payloads when the static leaf kernel reads them.
    C3_CUDA(cudaMallocManaged(&runtime.data_info, 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);  // logical tree base only
    runtime.data_info[2] = 2;  // L2
    C3_CUDA(cudaMallocManaged(&runtime.data_d, pool_values * sizeof(short)));
    for (std::size_t i = 0; i < pool_values; ++i) runtime.data_d[i] = static_cast<short>(pool[i]);
    indexConstru(runtime.data_d, nullptr, nullptr, runtime.data_info, runtime.id_list,
                 runtime.node_list, runtime.max_node_num, runtime.tree_height, runtime.empty_list);
    C3_CUDA(cudaDeviceSynchronize());
    C3_CUDA(cudaGetLastError());
    // indexConstru has already freed these temporaries internally.
    dis_list = nullptr; split_list = nullptr; split_num = nullptr; pid_list = nullptr;
    runtime.base_count = static_cast<int>(header.base_n);
    if (!runtime.ready()) fail("base tree construction did not produce a nontrivial runtime");

    const Snapshot initial = capture_snapshot(runtime);
    std::string search_upper_detail;
    const bool search_upper_invariant_ok = verify_search_upper_invariant(initial, &search_upper_detail);
    write_snapshot_json(args.initial_geometry, initial);
    std::ofstream output(args.output);
    if (!output) fail("cannot write engine JSONL");
    output << "{\"record\":\"meta\",\"schema\":\"c3-native-certified-direct-insert-runner-v1\","
           << "\"scope\":\"real GTS static top-K visibility after certified native leaf append; not complete C3\","
           << "\"archived_incremental_updater_used\":false,\"max_accepted\":" << args.max_accepted
           << ",\"strict_epsilon\":" << json_number(kStrictEpsilon)
           << ",\"max_size\":" << initial.max_size << ",\"leaf_pad_slots\":" << initial.leaf_pad_slots
           << ",\"search_upper_invariant_ok\":" << (search_upper_invariant_ok ? "true" : "false")
           << ",\"search_upper_detail\":\"" << search_upper_detail << "\"}\n";

    std::vector<std::uint8_t> active(header.pool_n, 0);
    std::fill(active.begin(), active.begin() + header.base_n, static_cast<std::uint8_t>(1));
    std::map<int, std::vector<int>> appended;
    for (int stable_id = static_cast<int>(header.base_n);
         stable_id < static_cast<int>(header.pool_n) && accepted < args.max_accepted; ++stable_id) {
      ++candidates;
      const Certificate certificate = certify_candidate(initial, pool, static_cast<int>(header.dimension),
                                                        stable_id, appended, search_upper_invariant_ok);
      if (!certificate.accepted) {
        emit_candidate(output, stable_id, certificate, -1, nullptr, nullptr);
        ++rejected;
        continue;
      }
      const Snapshot before_snapshot = capture_snapshot(runtime);
      const std::vector<int> leaf_before = before_snapshot.leaf_ids[certificate.leaf];
      if (static_cast<int>(leaf_before.size()) != certificate.pre_leaf_size) {
        fail("host leaf size disagrees with certificate before native append");
      }
      native_append(runtime, certificate.leaf, certificate.pre_leaf_size, stable_id);
      appended[certificate.leaf].push_back(stable_id);
      active[stable_id] = 1;
      const Snapshot after_snapshot = capture_snapshot(runtime);
      assert_native_layout(initial, after_snapshot, appended);
      const std::vector<int> leaf_after = after_snapshot.leaf_ids[certificate.leaf];
      if (leaf_after.size() != leaf_before.size() + 1 || leaf_after.back() != stable_id ||
          !std::equal(leaf_before.begin(), leaf_before.end(), leaf_after.begin())) {
        fail("target leaf append invariant failed");
      }
      emit_candidate(output, stable_id, certificate, initial.nodes[certificate.leaf].lid, &leaf_before, &leaf_after);

      // Actual GTS static API, not a host-side or sidecar answer.
      const GtsResult gts = run_real_static_topk(runtime, pool, static_cast<int>(header.dimension),
                                                 stable_id, static_cast<int>(header.k));
      const ExactResult exact = exact_topk(pool, static_cast<int>(header.dimension), stable_id,
                                           active, static_cast<int>(header.k));
      validate_real_query(gts, exact, stable_id);
      emit_self_query(output, stable_id, active_count(active), gts, exact);
      ++accepted;
      if (!output) fail("engine JSONL write failure");
    }
    const Snapshot final_snapshot = capture_snapshot(runtime);
    assert_native_layout(initial, final_snapshot, appended);
    write_snapshot_json(args.final_geometry, final_snapshot);
    output.close();
    if (accepted == 0) {
      write_summary(args.summary, "INCONCLUSIVE_NO_CERTIFIED_INSERT", "no candidate passed strict certificate/capacity gate",
                    &header, candidates, accepted, rejected, &plan);
      release_runtime(runtime);
      return 3;
    }
    write_summary(args.summary, "PASS_RUNNER_SELF_CHECKS_PENDING_CPU_VALIDATOR",
                  "all accepted native appends passed in-run geometry and real static self-query checks; independent CPU validator still required",
                  &header, candidates, accepted, rejected, &plan);
    release_runtime(runtime);
    return 0;
  } catch (const std::exception& error) {
    if (!args.summary.empty()) write_summary(args.summary, "FAIL_RUNNER_EXCEPTION", error.what(),
                                             header.base_n ? &header : nullptr, candidates, accepted, rejected,
                                             plan.infi_dis ? &plan : nullptr);
    std::cerr << error.what() << '\n';
    release_runtime(runtime);
    return 2;
  }
}
