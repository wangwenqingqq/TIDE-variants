// Isolated Safe-C1 G1 CUDA/GTS top-k visibility gate.
//
// This file deliberately does NOT include incremental_insert.cuh or update.cuh.
// It exposes only a frozen base-tree boundary and a stable-ID sidecar/delta model.
// Query merge kernels and trace I/O remain explicit TODOs; no legacy direct-insert
// path is considered evidence for E1-G.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "tree.cuh"
#include "safe_search_v2.cuh"

// `search_v2.cuh` uses these constants.  The Safe-C1 trace runner keeps mode=0
// unless a separate, held-out C2 experiment explicitly requests calibration.
void upload_rp_constants(float* h_alpha, float* h_beta, float* h_gamma,
                         int /*num_levels*/, float* h_lut_breaks,
                         float* h_lut_slopes, float* h_lut_intercepts,
                         int lut_size, int mode) {
  CHECK(cudaMemcpyToSymbol(c_rp_alpha, h_alpha,
                           RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_beta, h_beta,
                           RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_gamma, h_gamma,
                           RP_MAX_LEVELS * sizeof(float), 0,
                           cudaMemcpyHostToDevice));
  if (h_lut_breaks != nullptr && h_lut_slopes != nullptr &&
      h_lut_intercepts != nullptr) {
    CHECK(cudaMemcpyToSymbol(c_lut_breaks, h_lut_breaks,
                             RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_slopes, h_lut_slopes,
                             RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
    CHECK(cudaMemcpyToSymbol(c_lut_intercepts, h_lut_intercepts,
                             RP_LUT_SIZE * sizeof(float), 0,
                             cudaMemcpyHostToDevice));
  }
  CHECK(cudaMemcpyToSymbol(c_lut_num_segments, &lut_size, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
  CHECK(cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(int), 0,
                           cudaMemcpyHostToDevice));
}

namespace safe_c1 {

using StableId = int;
constexpr float kStrictEpsilon = 1.0e-5F;

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error("Safe-C1 invariant failure: " + message);
}

// The immutable pool is indexed by StableId.  `vectors` remains host-resident
// for certificate construction and the independent oracle; a production runner
// additionally owns a device copy that is intentionally distinct from the base
// tree's logical object count.
struct HostVectorPool {
  int dimension = 0;
  std::vector<float> vectors;

  [[nodiscard]] int size() const {
    return dimension == 0 ? 0 : static_cast<int>(vectors.size() / dimension);
  }

  [[nodiscard]] const float* at(StableId id) const {
    if (id < 0 || id >= size()) fail("stable ID outside immutable vector pool");
    return vectors.data() + static_cast<std::size_t>(id) * dimension;
  }

  [[nodiscard]] float l2(StableId left, StableId right) const {
    const float* a = at(left);
    const float* b = at(right);
    float sum = 0.0F;
    for (int j = 0; j < dimension; ++j) {
      const float diff = a[j] - b[j];
      sum += diff * diff;
    }
    return std::sqrt(sum);
  }
};

// Device buffers passed by the future trace loader.  At build time data_info[1]
// MUST equal base_count, while `data_d` may be allocated for the full immutable
// pool.  This is how the legacy tree is built only from base IDs while stable
// IDs continue to refer to the full external pool.
struct BaseTreeRuntime {
  int* data_info = nullptr;
  short* data_d = nullptr;
  char* data_s = nullptr;
  int* size_s = nullptr;
  int* id_list = nullptr;
  TN* node_list = nullptr;
  int* max_node_num = nullptr;
  int* empty_list = nullptr;
  int tree_height = 0;
  int base_count = 0;

  [[nodiscard]] bool ready() const {
    return data_info != nullptr && data_d != nullptr && node_list != nullptr &&
           id_list != nullptr && max_node_num != nullptr && empty_list != nullptr &&
           base_count > 0;
  }
};

// Host snapshot of exactly the data that Safe-C1 promises not to mutate between
// rebuilds.  Hashing detects accidental calls into the historical direct insert
// path or any mutation of interval/leaf layout.
struct FrozenTreeSnapshot {
  int tree_height = 0;
  int fanout = 0;
  std::vector<TN> nodes;
  std::vector<int> empty;
  std::vector<float> max_distance;
  // Hash the logical leaf payload as well as TN/interval metadata.  The legacy
  // id_list is padded, so we hash each active leaf's [node_id, size, ids]
  // layout rather than unknown padding capacity.
  std::uint64_t logical_leaf_id_hash = 0;
  int logical_leaf_id_count = 0;
  std::uint64_t hash = 0;

  [[nodiscard]] bool initialized() const { return !nodes.empty(); }

  static std::uint64_t fnv1a(const void* address, std::size_t bytes,
                             std::uint64_t seed = 1469598103934665603ULL) {
    const auto* ptr = static_cast<const unsigned char*>(address);
    std::uint64_t value = seed;
    for (std::size_t i = 0; i < bytes; ++i) {
      value ^= static_cast<std::uint64_t>(ptr[i]);
      value *= 1099511628211ULL;
    }
    return value;
  }

  [[nodiscard]] std::uint64_t compute_hash() const {
    std::uint64_t value = fnv1a(&tree_height, sizeof(tree_height));
    value = fnv1a(&fanout, sizeof(fanout), value);
    if (!nodes.empty()) value = fnv1a(nodes.data(), nodes.size() * sizeof(TN), value);
    if (!empty.empty()) value = fnv1a(empty.data(), empty.size() * sizeof(int), value);
    if (!max_distance.empty()) {
      value = fnv1a(max_distance.data(), max_distance.size() * sizeof(float), value);
    }
    value = fnv1a(&logical_leaf_id_count, sizeof(logical_leaf_id_count), value);
    value = fnv1a(&logical_leaf_id_hash, sizeof(logical_leaf_id_hash), value);
    return value;
  }

  void capture(const BaseTreeRuntime& runtime) {
    if (!runtime.ready()) fail("cannot snapshot an uninitialized base tree");
    const int count = runtime.max_node_num[0];
    if (count <= 0) fail("base tree has no nodes");
    tree_height = runtime.tree_height;
    fanout = TREE_ORDER;
    nodes.resize(count);
    empty.resize(count);
    max_distance.resize(count);
    CHECK(cudaMemcpy(nodes.data(), runtime.node_list, count * sizeof(TN),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(empty.data(), runtime.empty_list, count * sizeof(int),
                     cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(max_distance.data(), max_dis_d, count * sizeof(float),
                     cudaMemcpyDeviceToHost));
    std::vector<int> logical_layout;
    logical_layout.reserve(static_cast<std::size_t>(runtime.base_count) +
                           static_cast<std::size_t>(count) * 2U);
    logical_leaf_id_count = 0;
    for (int node_id = 0; node_id < count; ++node_id) {
      if (empty[node_id] != 0 || nodes[node_id].is_leaf != 1) continue;
      const TN& node = nodes[node_id];
      if (node.lid < 0 || node.size < 0) fail("invalid frozen leaf id-list range");
      logical_layout.push_back(node_id);
      logical_layout.push_back(node.size);
      if (node.size == 0) continue;
      std::vector<int> ids(static_cast<std::size_t>(node.size));
      CHECK(cudaMemcpy(ids.data(), runtime.id_list + node.lid,
                       ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
      logical_layout.insert(logical_layout.end(), ids.begin(), ids.end());
      if (logical_leaf_id_count > std::numeric_limits<int>::max() - node.size) {
        fail("frozen leaf id count overflow");
      }
      logical_leaf_id_count += node.size;
    }
    if (logical_leaf_id_count != runtime.base_count) {
      fail("frozen leaf payload does not cover exactly the base stable IDs");
    }
    logical_leaf_id_hash = fnv1a(logical_layout.data(),
                                 logical_layout.size() * sizeof(int));
    hash = compute_hash();
  }

  void assert_unchanged(const BaseTreeRuntime& runtime) const {
    FrozenTreeSnapshot current;
    current.capture(runtime);
    if (current.hash != hash) fail("frozen base-tree state changed outside rebuild");
  }

  [[nodiscard]] int only_strict_child(int parent, float distance) const {
    int match = -1;
    for (int slot = 0; slot < fanout; ++slot) {
      const int child = parent * fanout + slot + 1;
      if (child < 0 || child >= static_cast<int>(nodes.size()) || empty[child] != 0) {
        continue;
      }
      const float lower = nodes[child].min_dis;
      const float upper = max_distance[child];
      if (distance > lower + kStrictEpsilon && distance < upper - kStrictEpsilon) {
        if (match != -1) return -2;  // non-unique certificate
        match = child;
      }
    }
    return match;
  }

  // Returns a leaf only when every ancestor has exactly one strict interior
  // interval match.  A boundary, gap, overlap, or invalid tree falls back to
  // the global exact delta tier.
  [[nodiscard]] int certify_leaf(const HostVectorPool& pool,
                                 StableId inserted) const {
    if (!initialized() || tree_height <= 1 || fanout <= 1) return -1;
    int current = 0;
    for (int level = 0; level < tree_height - 1; ++level) {
      int pivot = -1;
      for (int slot = 0; slot < fanout; ++slot) {
        const int child = current * fanout + slot + 1;
        if (child >= 0 && child < static_cast<int>(nodes.size()) && empty[child] == 0) {
          pivot = nodes[child].pid;
          break;
        }
      }
      if (pivot < 0) return -1;
      const int child = only_strict_child(current, pool.l2(inserted, pivot));
      if (child < 0) return -1;
      if (nodes[child].is_leaf == 1) return child;
      current = child;
    }
    return nodes[current].is_leaf == 1 ? current : -1;
  }
};

enum class PlacementKind { kBase, kDirect, kDelta, kDeleted, kUnknown };

struct Placement {
  PlacementKind kind = PlacementKind::kUnknown;
  int leaf_id = -1;
};

// Pure host state: accepted direct IDs live outside legacy `id_list`; rejected
// IDs live in an exact global delta.  No method mutates a legacy TN or id_list.
class SafeC1State {
 public:
  SafeC1State(int pool_size, int leaf_capacity)
      : active_(pool_size, 0), leaf_capacity_(leaf_capacity) {
    if (pool_size <= 0 || leaf_capacity <= 0) fail("invalid Safe-C1 capacity");
  }

  void initialize_base(const std::vector<StableId>& base_ids) {
    for (StableId id : base_ids) {
      check_id(id);
      if (active_[id] != 0) fail("duplicate base stable ID");
      active_[id] = 1;
      placement_[id] = Placement{PlacementKind::kBase, -1};
    }
  }

  [[nodiscard]] Placement insert(StableId id, const FrozenTreeSnapshot& frozen,
                                 const HostVectorPool& pool) {
    check_id(id);
    if (active_[id] != 0) fail("inserted stable ID is already active");
    active_[id] = 1;
    const int leaf = frozen.certify_leaf(pool, id);
    if (leaf >= 0 && static_cast<int>(sidecars_[leaf].size()) < leaf_capacity_) {
      sidecars_[leaf].push_back(id);
      placement_[id] = Placement{PlacementKind::kDirect, leaf};
      return placement_[id];
    }
    delta_.push_back(id);
    placement_[id] = Placement{PlacementKind::kDelta, -1};
    return placement_[id];
  }

  void erase(StableId id) {
    check_id(id);
    if (active_[id] == 0) fail("delete targets inactive stable ID");
    active_[id] = 0;
    const auto it = placement_.find(id);
    if (it == placement_.end()) fail("active ID missing placement metadata");
    if (it->second.kind == PlacementKind::kDirect) {
      auto& values = sidecars_[it->second.leaf_id];
      values.erase(std::remove(values.begin(), values.end(), id), values.end());
    } else if (it->second.kind == PlacementKind::kDelta) {
      delta_.erase(std::remove(delta_.begin(), delta_.end(), id), delta_.end());
    }
    it->second = Placement{PlacementKind::kDeleted, -1};
  }

  // Called only immediately after an explicitly logged rebuild.  The caller
  // gives every live stable ID that is now physically represented by the fresh
  // base tree; direct/delta tiers then become empty.
  void after_rebuild(const std::vector<StableId>& live_base_ids) {
    std::fill(active_.begin(), active_.end(), 0);
    sidecars_.clear();
    delta_.clear();
    placement_.clear();
    initialize_base(live_base_ids);
  }

  [[nodiscard]] const std::unordered_map<int, std::vector<StableId>>& sidecars() const {
    return sidecars_;
  }
  [[nodiscard]] const std::vector<StableId>& delta() const { return delta_; }
  [[nodiscard]] bool is_active(StableId id) const {
    check_id(id);
    return active_[id] != 0;
  }
  [[nodiscard]] std::uint64_t active_hash() const {
    return FrozenTreeSnapshot::fnv1a(active_.data(), active_.size());
  }

 private:
  void check_id(StableId id) const {
    if (id < 0 || id >= static_cast<StableId>(active_.size())) {
      fail("stable ID outside configured pool");
    }
  }

  std::vector<std::uint8_t> active_;
  int leaf_capacity_;
  std::unordered_map<int, std::vector<StableId>> sidecars_;
  std::vector<StableId> delta_;
  std::unordered_map<StableId, Placement> placement_;
};

struct TopKPairs {
  std::vector<StableId> ids;
  std::vector<float> distances;
};

// Receipt emitted by the copied, real GTS vector-KNN traversal. The leaf IDs
// originate at mergeLNodeKnn before native leaf scanning; they are not inferred
// from Safe-C1 certificates. The G1 query layer may scan sidecars only from
// this receipt, while delta remains an explicit global exact tier.
struct TopKWithTraversalReceipt {
  TopKPairs topk;
  std::vector<int> visited_leaf_ids;
};

TopKWithTraversalReceipt run_gts_base_topk_with_receipt(
    BaseTreeRuntime& runtime, float* query_vectors, int qnum, int k) {
  if (!runtime.ready() || query_vectors == nullptr || qnum <= 0 || k <= 0) {
    fail("invalid GTS top-k arguments");
  }
  int* result_ids = nullptr;
  CHECK(cudaMallocManaged((void**)&result_ids,
                           static_cast<std::size_t>(qnum) * k * sizeof(int)));
  update_disk = false;
  searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list,
                    runtime.max_node_num, query_vectors, result_ids, qnum, k,
                    runtime.tree_height, runtime.data_info, runtime.empty_list,
                    runtime.data_s, runtime.size_s);
  CHECK(cudaDeviceSynchronize());

  TopKWithTraversalReceipt output;
  output.topk.ids.assign(result_ids,
                         result_ids + static_cast<std::size_t>(qnum) * k);
  output.topk.distances.assign(res_dis,
                               res_dis + static_cast<std::size_t>(qnum) * k);
  for (const SafeC1VisitedLeafPair& pair : safe_c1_last_visited_leaf_pairs) {
    if (pair.query_id < 0 || pair.query_id >= qnum || pair.leaf_id < 0) {
      CHECK(cudaFree(result_ids));
      CHECK(cudaFree(res_dis));
      res_dis = nullptr;
      fail("invalid receipt emitted by copied GTS traversal");
    }
    output.visited_leaf_ids.push_back(pair.leaf_id);
  }
  std::sort(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end());
  output.visited_leaf_ids.erase(
      std::unique(output.visited_leaf_ids.begin(), output.visited_leaf_ids.end()),
      output.visited_leaf_ids.end());
  CHECK(cudaFree(result_ids));
  CHECK(cudaFree(res_dis));
  res_dis = nullptr;
  return output;
}

TopKPairs probe_static_base_topk(BaseTreeRuntime& runtime, float* query_vectors,
                                int qnum, int k) {
  return run_gts_base_topk_with_receipt(runtime, query_vectors, qnum, k).topk;
}

struct SafeQueryApi {
  [[nodiscard]] static const char* range_status() {
    return "G1 deliberately top-k-only: stable-ID range export remains unimplemented";
  }
  [[nodiscard]] static const char* topk_status() {
    return "G1 implemented: real GTS base traversal receipt + visited-leaf sidecars + global delta merge";
  }
};

}  // namespace safe_c1

// -----------------------------------------------------------------------------
// E1-GI-B trace executor
// -----------------------------------------------------------------------------
// The archived mutable update implementation is deliberately never called.
// The legacy GTS tree is built from the immutable base only.  Its real static
// vector top-k API is probed and checked against an exact base oracle before
// tracing.  Safe-C1 direct/delta state is then maintained outside the tree; the
// dynamic answer exporter uses an exact CUDA scan of the active stable-ID pool.
// This is a correctness integration, not a selective-query latency result.

namespace {

constexpr char kTraceMagic[8] = {'E', '1', 'G', 'T', 'R', 'C', '0', '1'};
constexpr std::uint32_t kTraceVersion = 1;
constexpr float kProbeDistanceAbsoluteFloor = 2.0e-3F;
constexpr int kProbeDistanceMaxFloatUlps = 2;

enum TraceOp : std::uint8_t { kInsert = 1, kDelete = 2, kKnn = 3, kRange = 4 };

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
struct TraceEvent {
  std::uint32_t op_index;
  std::uint8_t op;
  std::uint8_t reserved[3];
  std::int32_t argument;
};
#pragma pack(pop)
static_assert(sizeof(TraceHeader) == 48, "unexpected trace header ABI");
static_assert(sizeof(TraceEvent) == 12, "unexpected trace event ABI");

struct Args {
  std::string bundle;
  std::string output;
  std::string summary;
  int leaf_capacity = 64;
};

[[noreturn]] void die(const std::string& message) {
  throw std::runtime_error("Safe-C1 G1: " + message);
}

void cuda_or_die(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << "CUDA " << expression << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(status);
  die(out.str());
}
#define GI_CUDA(expr) cuda_or_die((expr), #expr, __FILE__, __LINE__)

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key(argv[i]);
    auto value = [&]() -> std::string {
      if (++i >= argc) die("missing value after " + key);
      return argv[i];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--out") args.output = value();
    else if (key == "--summary") args.summary = value();
    else if (key == "--leaf-capacity") args.leaf_capacity = std::stoi(value());
    else if (key == "--help") {
      std::cout << "usage: GTS_safe_c1_g1 --bundle QUANTIZED_BUNDLE --out RESULTS.jsonl "
                   "--summary SUMMARY.json [--leaf-capacity 2]\n";
      std::exit(0);
    } else {
      die("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.output.empty() || args.summary.empty()) {
    die("--bundle, --out, and --summary are required");
  }
  if (args.leaf_capacity <= 0) die("leaf capacity must be positive");
  return args;
}

template <class T>
std::vector<T> read_exact_binary(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) die("cannot open " + path);
  std::vector<T> out(count);
  input.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(T))) {
    die("unexpected size for " + path);
  }
  char extra = 0;
  if (input.read(&extra, 1)) die("extra bytes in " + path);
  return out;
}

std::pair<TraceHeader, std::vector<TraceEvent>> read_trace(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) die("cannot open trace " + path);
  TraceHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) die("truncated trace header");
  if (std::memcmp(header.magic, kTraceMagic, sizeof(kTraceMagic)) != 0 ||
      header.version != kTraceVersion) {
    die("unsupported trace magic/version");
  }
  if (header.dimension == 0 || header.base_n == 0 || header.pool_n < header.base_n ||
      header.reservoir_n != header.pool_n - header.base_n || header.query_n == 0 ||
      header.k == 0 || header.k > header.base_n || header.radius < 0.0F) {
    die("invalid trace header");
  }
  std::vector<TraceEvent> events(header.event_count);
  input.read(reinterpret_cast<char*>(events.data()),
             static_cast<std::streamsize>(events.size() * sizeof(TraceEvent)));
  if (input.gcount() != static_cast<std::streamsize>(events.size() * sizeof(TraceEvent))) {
    die("truncated trace event payload");
  }
  char extra = 0;
  if (input.read(&extra, 1)) die("extra bytes in trace");
  for (std::size_t index = 0; index < events.size(); ++index) {
    const TraceEvent& event = events[index];
    if (event.op_index != index || event.op < kInsert || event.op > kRange) {
      die("invalid non-contiguous trace event at " + std::to_string(index));
    }
  }
  return {header, std::move(events)};
}

// GTS uses INFI_DIS both as the initial top-k sentinel and to encode a local
// distance inside a query/pivot sort key.  The historical fixed value (10000)
// is invalid for a quantized dataset whenever a true L2 distance can exceed it:
// candidates would be replaced by the sentinel before ranking.  Compute a
// conservative bound from the coordinates actually supplied to this isolated
// run.  Every pool/query pair lies in the per-dimension bounding box, and the
// extra 1024 units absorb float accumulation/sqrt rounding while retaining a
// strict sentinel upper bound.  This changes no original archive/config file.
struct MetricEncodingPlan {
  std::uint64_t bbox_squared_upper_bound = 0;
  long double bbox_l2_upper_bound = 0.0L;
  int infi_dis = 0;
  int dis_code = 100;
  bool raised_above_legacy_default = false;
};

MetricEncodingPlan make_metric_encoding_plan(
    const std::vector<std::int16_t>& pool_i16,
    const std::vector<std::int16_t>& queries_i16, int dimension) {
  constexpr int kLegacyInfiDis = 10000;
  constexpr int kDisCode = 100;
  constexpr long double kFloatSafetyMargin = 1024.0L;
  if (dimension <= 0 || pool_i16.empty() || queries_i16.empty() ||
      pool_i16.size() % static_cast<std::size_t>(dimension) != 0 ||
      queries_i16.size() % static_cast<std::size_t>(dimension) != 0) {
    die("cannot derive metric encoding bound from malformed quantized inputs");
  }
  std::vector<int> lower(dimension, std::numeric_limits<int>::max());
  std::vector<int> upper(dimension, std::numeric_limits<int>::min());
  const auto incorporate = [&](const std::vector<std::int16_t>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      const int coordinate = static_cast<int>(values[index]);
      const int axis = static_cast<int>(index % static_cast<std::size_t>(dimension));
      lower[axis] = std::min(lower[axis], coordinate);
      upper[axis] = std::max(upper[axis], coordinate);
    }
  };
  incorporate(pool_i16);
  incorporate(queries_i16);

  std::uint64_t squared_bound = 0;
  for (int axis = 0; axis < dimension; ++axis) {
    const std::int64_t span = static_cast<std::int64_t>(upper[axis]) - lower[axis];
    const std::uint64_t term = static_cast<std::uint64_t>(span * span);
    if (squared_bound > std::numeric_limits<std::uint64_t>::max() - term) {
      die("metric bounding-box square overflow");
    }
    squared_bound += term;
  }
  const long double l2_bound = std::sqrt(static_cast<long double>(squared_bound));
  const long double required = std::ceil(l2_bound) + kFloatSafetyMargin;
  if (!std::isfinite(required) ||
      required > static_cast<long double>(std::numeric_limits<int>::max())) {
    die("metric encoding bound cannot fit GTS INFI_DIS int");
  }
  MetricEncodingPlan plan;
  plan.bbox_squared_upper_bound = squared_bound;
  plan.bbox_l2_upper_bound = l2_bound;
  plan.infi_dis = std::max(kLegacyInfiDis, static_cast<int>(required));
  plan.dis_code = kDisCode;
  plan.raised_above_legacy_default = plan.infi_dis > kLegacyInfiDis;
  // `disk` is float in the legacy static search path, so keep the sentinel an
  // exactly representable integer rather than silently rounding it down.
  constexpr int kMaxExactFloatInteger = 1 << 24;
  if (plan.infi_dis > kMaxExactFloatInteger) {
    die("metric encoding sentinel exceeds exact float32 integer range");
  }
  if (!(static_cast<long double>(plan.infi_dis) > l2_bound) || plan.dis_code <= 1) {
    die("invalid metric encoding plan");
  }
  return plan;
}

class IntegrationState {
 public:
  IntegrationState(int pool_n, int base_n, int leaf_capacity)
      : active_(pool_n, 0), placement_(pool_n, safe_c1::PlacementKind::kDeleted),
        leaf_for_(pool_n, -1), leaf_capacity_(leaf_capacity) {
    if (pool_n <= 0 || base_n <= 0 || base_n > pool_n) die("bad state dimensions");
    for (int id = 0; id < base_n; ++id) {
      active_[id] = 1;
      placement_[id] = safe_c1::PlacementKind::kBase;
    }
  }

  safe_c1::PlacementKind insert(int id, const safe_c1::FrozenTreeSnapshot& frozen,
                                const safe_c1::HostVectorPool& pool) {
    check(id);
    if (active_[id] != 0) die("insert targets active stable ID " + std::to_string(id));
    active_[id] = 1;
    const int leaf = frozen.certify_leaf(pool, id);
    if (leaf >= 0 && static_cast<int>(sidecars_[leaf].size()) < leaf_capacity_) {
      sidecars_[leaf].push_back(id);
      placement_[id] = safe_c1::PlacementKind::kDirect;
      leaf_for_[id] = leaf;
      ++direct_inserts_;
      return placement_[id];
    }
    placement_[id] = safe_c1::PlacementKind::kDelta;
    delta_.push_back(id);
    leaf_for_[id] = -1;
    ++delta_inserts_;
    if (leaf < 0) ++certificate_rejects_;
    else ++capacity_rejects_;
    return placement_[id];
  }

  void erase(int id) {
    check(id);
    if (active_[id] == 0) die("delete targets inactive stable ID " + std::to_string(id));
    const safe_c1::PlacementKind prior = placement_[id];
    if (prior == safe_c1::PlacementKind::kDirect) {
      const int leaf = leaf_for_[id];
      auto it = sidecars_.find(leaf);
      if (it == sidecars_.end()) die("missing direct sidecar placement");
      auto& values = it->second;
      values.erase(std::remove(values.begin(), values.end(), id), values.end());
    } else if (prior == safe_c1::PlacementKind::kDelta) {
      delta_.erase(std::remove(delta_.begin(), delta_.end(), id), delta_.end());
    }
    active_[id] = 0;
    placement_[id] = safe_c1::PlacementKind::kDeleted;
    leaf_for_[id] = -1;
  }

  const std::vector<std::uint8_t>& active() const { return active_; }
  int direct_inserts() const { return direct_inserts_; }
  int delta_inserts() const { return delta_inserts_; }
  int certificate_rejects() const { return certificate_rejects_; }
  int capacity_rejects() const { return capacity_rejects_; }
  int active_count() const {
    return static_cast<int>(std::count(active_.begin(), active_.end(), static_cast<std::uint8_t>(1)));
  }
  int direct_live() const {
    return static_cast<int>(std::count(placement_.begin(), placement_.end(), safe_c1::PlacementKind::kDirect));
  }
  int delta_live() const {
    return static_cast<int>(std::count(placement_.begin(), placement_.end(), safe_c1::PlacementKind::kDelta));
  }
  safe_c1::PlacementKind placement_of(int id) const {
    check(id);
    return placement_[id];
  }
  int leaf_of(int id) const {
    check(id);
    return leaf_for_[id];
  }
  std::vector<int> sidecar_candidates_for(const std::vector<int>& visited_leaf_ids) const {
    std::vector<int> candidates;
    for (int leaf : visited_leaf_ids) {
      const auto it = sidecars_.find(leaf);
      if (it == sidecars_.end()) continue;
      for (int id : it->second) {
        if (active_[id] != 0 && placement_[id] == safe_c1::PlacementKind::kDirect) {
          candidates.push_back(id);
        }
      }
    }
    std::sort(candidates.begin(), candidates.end());
    candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
    return candidates;
  }
  const std::vector<int>& delta_ids() const { return delta_; }

 private:
  void check(int id) const {
    if (id < 0 || id >= static_cast<int>(active_.size())) die("stable ID outside pool");
  }
  std::vector<std::uint8_t> active_;
  std::vector<safe_c1::PlacementKind> placement_;
  std::vector<int> leaf_for_;
  int leaf_capacity_;
  std::unordered_map<int, std::vector<int>> sidecars_;
  std::vector<int> delta_;
  int direct_inserts_ = 0;
  int delta_inserts_ = 0;
  int certificate_rejects_ = 0;
  int capacity_rejects_ = 0;
};

__global__ void exact_candidate_l2(const float* pool, const float* query,
                                  const int* candidate_ids, double* distances,
                                  int candidate_count, int dimension) {
  const int position = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (position >= candidate_count) return;
  const int stable_id = candidate_ids[position];
  const float* vector = pool + static_cast<std::size_t>(stable_id) * dimension;
  double sum = 0.0;
  for (int d = 0; d < dimension; ++d) {
    const double delta = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
    sum += delta * delta;
  }
  distances[position] = sqrt(sum);
}

std::vector<std::pair<float, int>> exact_host(const safe_c1::HostVectorPool& pool,
                                                const float* query,
                                                const std::vector<std::uint8_t>& active) {
  std::vector<std::pair<float, int>> rows;
  rows.reserve(pool.size());
  for (int id = 0; id < pool.size(); ++id) {
    if (active[id] == 0) continue;
    const float* vector = pool.at(id);
    double sum = 0.0;
    for (int d = 0; d < pool.dimension; ++d) {
      const double delta = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
      sum += delta * delta;
    }
    rows.emplace_back(static_cast<float>(std::sqrt(sum)), id);
  }
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.first != right.first ? left.first < right.first : left.second < right.second;
  });
  return rows;
}

bool close_distance(float left, float right) {
  // The real GTS static path accumulates/square-roots in float32, while the
  // host reference computes L2 in float64 and then materializes float32.  IDs
  // still require an exact positional match; distances may differ only by the
  // documented two-float-ULP envelope (or the legacy tiny absolute floor).
  const float magnitude = std::max({1.0F, std::fabs(left), std::fabs(right)});
  const float ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
  const float tolerance = std::max(
      kProbeDistanceAbsoluteFloor,
      static_cast<float>(kProbeDistanceMaxFloatUlps) * ulp);
  return std::fabs(left - right) <= tolerance;
}

const char* placement_label(safe_c1::PlacementKind placement) {
  switch (placement) {
    case safe_c1::PlacementKind::kBase: return "base";
    case safe_c1::PlacementKind::kDirect: return "direct";
    case safe_c1::PlacementKind::kDelta: return "delta";
    case safe_c1::PlacementKind::kDeleted: return "deleted";
    default: return "unknown";
  }
}

std::string json_float(double value) {
  std::ostringstream out;
  out << std::setprecision(9) << value;
  return out.str();
}

void emit_int_array(std::ofstream& output, const std::vector<int>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
}

void emit_update(std::ofstream& output, const TraceEvent& event,
                 safe_c1::PlacementKind placement, int leaf_id) {
  output << "{\"record\":\"update\",\"op_index\":" << event.op_index
         << ",\"op\":\"" << (event.op == kInsert ? "insert" : "delete")
         << "\",\"stable_id\":" << event.argument
         << ",\"placement\":\"" << placement_label(placement) << "\""
         << ",\"sidecar_leaf_id\":" << leaf_id << "}\n";
}

void emit_query(std::ofstream& output, const TraceEvent& event,
                const std::vector<std::pair<double, int>>& rows,
                const std::vector<int>& visited_leaf_ids,
                int sidecar_candidate_count, int delta_candidate_count) {
  output << "{\"record\":\"query\",\"op_index\":" << event.op_index
         << ",\"kind\":\"knn\",\"query_id\":" << event.argument
         << ",\"gts_visited_leaf_ids\":";
  emit_int_array(output, visited_leaf_ids);
  output << ",\"sidecar_candidate_count\":" << sidecar_candidate_count
         << ",\"delta_candidate_count\":" << delta_candidate_count
         << ",\"results\":[";
  for (std::size_t index = 0; index < rows.size(); ++index) {
    if (index) output << ',';
    output << '[' << rows[index].second << ',' << json_float(rows[index].first) << ']';
  }
  output << "]}\n";
}

void release_tree(safe_c1::BaseTreeRuntime& runtime) {
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

}  // namespace

int main(int argc, char** argv) {
  safe_c1::BaseTreeRuntime runtime;
  float* query_d = nullptr;
  float* full_pool_d = nullptr;
  int* candidate_ids_d = nullptr;
  double* candidate_distances_d = nullptr;
  try {
    const Args args = parse_args(argc, argv);
    const auto [header, events] = read_trace(args.bundle + "/trace.e1gtrc");
    const std::size_t pool_values = static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values = static_cast<std::size_t>(header.query_n) * header.dimension;
    const std::vector<std::int16_t> pool_i16 = read_exact_binary<std::int16_t>(args.bundle + "/pool.i16", pool_values);
    const std::vector<std::int16_t> queries_i16 = read_exact_binary<std::int16_t>(args.bundle + "/queries.i16", query_values);

    safe_c1::HostVectorPool pool;
    pool.dimension = static_cast<int>(header.dimension);
    pool.vectors.resize(pool_values);
    for (std::size_t index = 0; index < pool_i16.size(); ++index) pool.vectors[index] = static_cast<float>(pool_i16[index]);
    std::vector<float> query_host(query_values);
    for (std::size_t index = 0; index < queries_i16.size(); ++index) query_host[index] = static_cast<float>(queries_i16[index]);

    // Configure only this isolated process before the first GTS kernel.  The
    // tree construction keeps its metric/routing semantics: within each pivot
    // block the positive normalization is order-preserving, while all encoded
    // distances remain strictly below one and therefore below DIS_CODE.
    const MetricEncodingPlan metric_encoding = make_metric_encoding_plan(
        pool_i16, queries_i16, static_cast<int>(header.dimension));
    DIS_CODE = metric_encoding.dis_code;
    INFI_DIS = metric_encoding.infi_dis;
    std::cout << "Safe-C1 G1 metric_encoding infi_dis=" << INFI_DIS
              << " dis_code=" << DIS_CODE
              << " bbox_l2_upper_bound=" << static_cast<double>(metric_encoding.bbox_l2_upper_bound)
              << '\n';

    // The base tree sees exactly [0, base_n), never reservoir rows.
    GI_CUDA(cudaMallocManaged(&runtime.data_info, 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);
    runtime.data_info[2] = 2;  // L2.
    GI_CUDA(cudaMallocManaged(&runtime.data_d,
                              static_cast<std::size_t>(header.base_n) * header.dimension * sizeof(short)));
    for (std::size_t index = 0; index < static_cast<std::size_t>(header.base_n) * header.dimension; ++index) {
      runtime.data_d[index] = static_cast<short>(pool_i16[index]);
    }
    indexConstru(runtime.data_d, nullptr, nullptr, runtime.data_info, runtime.id_list,
                 runtime.node_list, runtime.max_node_num, runtime.tree_height, runtime.empty_list);
    // The archived constructor sometimes only prints CUDA failures.  Surface
    // them at the harness boundary, then null its temporary globals because it
    // has already freed them internally (avoid release_tree double-free).
    GI_CUDA(cudaDeviceSynchronize());
    GI_CUDA(cudaGetLastError());
    dis_list = nullptr;
    split_list = nullptr;
    split_num = nullptr;
    pid_list = nullptr;
    runtime.base_count = static_cast<int>(header.base_n);
    if (!runtime.ready()) die("GTS base tree construction did not produce a runtime");

    // Baseline residual-pruning mode: no learned pruning and no calibration split.
    float alpha[RP_MAX_LEVELS]{};
    float beta[RP_MAX_LEVELS]{};
    float gamma[RP_MAX_LEVELS];
    std::fill(std::begin(gamma), std::end(gamma), 1.0F);
    upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, 0);

    safe_c1::FrozenTreeSnapshot frozen;
    frozen.capture(runtime);

    GI_CUDA(cudaMallocManaged(&query_d, query_values * sizeof(float)));
    std::copy(query_host.begin(), query_host.end(), query_d);

    // Real GTS static top-k API probe across all external queries.  This is
    // intentionally separate from the exact dynamic exporter below.
    const safe_c1::TopKPairs gts_probe = safe_c1::probe_static_base_topk(
        runtime, query_d, static_cast<int>(header.query_n), static_cast<int>(header.k));
    // Top-k membership is exact on untied ranks.  GTS exposes no documented
    // stable-ID tie-break, so within an equal-distance group wholly inside the
    // returned top-k we compare the ID set, not incidental output order.  A
    // tie at the k-th boundary remains an explicit blocker because its top-k
    // membership is not uniquely defined.
    int gts_probe_id_mismatch = 0;
    int gts_probe_distance_mismatch = 0;
    int gts_probe_positional_tie_reorders = 0;
    int gts_probe_internal_tie_groups = 0;
    double gts_probe_max_abs_distance_delta = 0.0;
    double gts_probe_max_relative_distance_delta = 0.0;
    std::ostringstream gts_probe_distance_examples;
    std::ostringstream gts_probe_id_examples;
    int gts_probe_distance_examples_count = 0;
    int gts_probe_id_examples_count = 0;
    int gts_probe_ambiguous_boundary_ties = 0;
    std::vector<std::uint8_t> base_active(header.pool_n, 0);
    std::fill(base_active.begin(), base_active.begin() + header.base_n, 1);
    for (int qid = 0; qid < static_cast<int>(header.query_n); ++qid) {
      std::vector<std::pair<float, int>> expected = exact_host(
          pool, query_host.data() + static_cast<std::size_t>(qid) * header.dimension, base_active);
      if (expected.size() < header.k) die("base oracle has fewer than k candidates");
      if (expected.size() > header.k &&
          expected[header.k - 1].first == expected[header.k].first) {
        ++gts_probe_ambiguous_boundary_ties;
      }
      expected.resize(header.k);
      std::vector<int> observed_ids(header.k);
      for (int position = 0; position < static_cast<int>(header.k); ++position) {
        const std::size_t offset = static_cast<std::size_t>(qid) * header.k + position;
        observed_ids[position] = gts_probe.ids[offset];
        const float observed = gts_probe.distances[offset];
        const float reference = expected[position].first;
        const double absolute_delta = std::fabs(static_cast<double>(observed) - static_cast<double>(reference));
        const double relative_delta = absolute_delta / std::max(1.0, std::fabs(static_cast<double>(reference)));
        gts_probe_max_abs_distance_delta = std::max(gts_probe_max_abs_distance_delta, absolute_delta);
        gts_probe_max_relative_distance_delta = std::max(gts_probe_max_relative_distance_delta, relative_delta);
        if (!close_distance(observed, reference)) {
          ++gts_probe_distance_mismatch;
          if (gts_probe_distance_examples_count < 5) {
            if (gts_probe_distance_examples_count != 0) gts_probe_distance_examples << ';';
            gts_probe_distance_examples << "q=" << qid << ",p=" << position
                                        << ",obs=" << std::setprecision(9) << observed
                                        << ",ref=" << reference
                                        << ",abs=" << absolute_delta
                                        << ",rel=" << relative_delta;
            ++gts_probe_distance_examples_count;
          }
        }
      }
      for (int begin = 0; begin < static_cast<int>(header.k);) {
        int end = begin + 1;
        while (end < static_cast<int>(header.k) &&
               expected[end].first == expected[begin].first) {
          ++end;
        }
        std::vector<int> expected_ids;
        std::vector<int> actual_ids;
        expected_ids.reserve(static_cast<std::size_t>(end - begin));
        actual_ids.reserve(static_cast<std::size_t>(end - begin));
        for (int position = begin; position < end; ++position) {
          expected_ids.push_back(expected[position].second);
          actual_ids.push_back(observed_ids[position]);
          if (observed_ids[position] != expected[position].second) {
            ++gts_probe_positional_tie_reorders;
          }
        }
        if (end - begin > 1) ++gts_probe_internal_tie_groups;
        std::sort(expected_ids.begin(), expected_ids.end());
        std::sort(actual_ids.begin(), actual_ids.end());
        if (expected_ids != actual_ids) {
          gts_probe_id_mismatch += end - begin;
          if (gts_probe_id_examples_count < 5) {
            if (gts_probe_id_examples_count != 0) gts_probe_id_examples << ';';
            gts_probe_id_examples << "q=" << qid << ",range=" << begin << '-' << (end - 1)
                                  << ",expected=";
            for (std::size_t index = 0; index < expected_ids.size(); ++index) {
              if (index) gts_probe_id_examples << ',';
              gts_probe_id_examples << expected_ids[index];
            }
            gts_probe_id_examples << ",actual=";
            for (std::size_t index = 0; index < actual_ids.size(); ++index) {
              if (index) gts_probe_id_examples << ',';
              gts_probe_id_examples << actual_ids[index];
            }
            ++gts_probe_id_examples_count;
          }
        }
        begin = end;
      }
    }
    if (gts_probe_id_mismatch != 0 || gts_probe_distance_mismatch != 0 ||
        gts_probe_ambiguous_boundary_ties != 0) {
      std::ostringstream detail;
      detail << "real GTS static base probe mismatch: id_membership=" << gts_probe_id_mismatch
             << " distance=" << gts_probe_distance_mismatch
             << " ambiguous_boundary_ties=" << gts_probe_ambiguous_boundary_ties
             << " positional_tie_reorders=" << gts_probe_positional_tie_reorders
             << " max_abs=" << std::setprecision(9) << gts_probe_max_abs_distance_delta
             << " max_rel=" << gts_probe_max_relative_distance_delta
             << " distance_examples=" << gts_probe_distance_examples.str()
             << " id_examples=" << gts_probe_id_examples.str();
      die(detail.str());
    }
    frozen.assert_unchanged(runtime);

    // G1 scope: frozen base tree, strict certificate sidecars, exact global delta,
    // and top-k only.  The first gate deliberately rejects base deletion (which
    // requires an explicit compacting rebuild) and range queries (whose legacy
    // GTS API cannot export stable IDs).  It never invokes legacy update code.
    IntegrationState state(static_cast<int>(header.pool_n), static_cast<int>(header.base_n),
                           args.leaf_capacity);
    GI_CUDA(cudaMalloc(&full_pool_d, pool.vectors.size() * sizeof(float)));
    GI_CUDA(cudaMalloc(&candidate_ids_d, static_cast<std::size_t>(header.pool_n) * sizeof(int)));
    GI_CUDA(cudaMalloc(&candidate_distances_d,
                       static_cast<std::size_t>(header.pool_n) * sizeof(double)));
    GI_CUDA(cudaMemcpy(full_pool_d, pool.vectors.data(),
                       pool.vectors.size() * sizeof(float), cudaMemcpyHostToDevice));

    std::ofstream output(args.output);
    if (!output) die("cannot write result JSONL");
    output << "{\"record\":\"meta\",\"schema\":\"safe-c1-g1-topk-v1\","
           << "\"scope\":\"G1 correctness only: real GTS base traversal receipt; visited-leaf-only sidecars; exact global delta; no base delete, rebuild, range, or latency claim\","
           << "\"gts_static_probe\":\"PASS\"}\n";

    int inserts = 0, deletes = 0, knn = 0;
    std::uint64_t sidecar_candidate_distances = 0;
    std::uint64_t delta_candidate_distances = 0;
    std::uint64_t visited_leaf_total = 0;
    int query_with_sidecar_candidates = 0;
    for (const TraceEvent& event : events) {
      const int argument = event.argument;
      if (event.op == kInsert) {
        if (argument < static_cast<int>(header.base_n) || argument >= static_cast<int>(header.pool_n)) {
          die("insert does not address disjoint reservoir at op " + std::to_string(event.op_index));
        }
        const safe_c1::PlacementKind placement = state.insert(argument, frozen, pool);
        frozen.assert_unchanged(runtime);
        emit_update(output, event, placement, state.leaf_of(argument));
        ++inserts;
      } else if (event.op == kDelete) {
        if (argument < 0 || argument >= static_cast<int>(header.pool_n)) {
          die("delete stable ID out of range");
        }
        const safe_c1::PlacementKind prior = state.placement_of(argument);
        if (prior == safe_c1::PlacementKind::kBase) {
          die("G1 refuses base deletion: compacting rebuild is an explicit G2 requirement");
        }
        const int prior_leaf = state.leaf_of(argument);
        state.erase(argument);
        frozen.assert_unchanged(runtime);
        emit_update(output, event, safe_c1::PlacementKind::kDeleted, prior_leaf);
        ++deletes;
      } else if (event.op == kKnn) {
        if (argument < 0 || argument >= static_cast<int>(header.query_n)) {
          die("query ID out of range");
        }
        float* query = query_d + static_cast<std::size_t>(argument) * header.dimension;

        // This call executes the copied real GTS vector-KNN traversal.  Its
        // receipt is the sole authority for which sidecar leaves are scanned.
        const safe_c1::TopKWithTraversalReceipt gts_answer =
            safe_c1::run_gts_base_topk_with_receipt(runtime, query, 1,
                                                    static_cast<int>(header.k));
        if (gts_answer.topk.ids.size() != header.k ||
            gts_answer.topk.distances.size() != header.k) {
          die("GTS top-k returned an unexpected candidate count");
        }
        frozen.assert_unchanged(runtime);

        std::vector<int> sidecar_ids = state.sidecar_candidates_for(gts_answer.visited_leaf_ids);
        const std::vector<int>& delta_ids = state.delta_ids();
        std::vector<int> extra_ids;
        extra_ids.reserve(sidecar_ids.size() + delta_ids.size());
        extra_ids.insert(extra_ids.end(), sidecar_ids.begin(), sidecar_ids.end());
        extra_ids.insert(extra_ids.end(), delta_ids.begin(), delta_ids.end());
        std::sort(extra_ids.begin(), extra_ids.end());
        if (std::adjacent_find(extra_ids.begin(), extra_ids.end()) != extra_ids.end()) {
          die("Safe-C1 sidecar/delta candidate IDs overlap");
        }
        for (int id : extra_ids) {
          if (id < static_cast<int>(header.base_n) || state.active()[id] == 0) {
            die("invalid inactive/base ID in Safe-C1 extra candidate tier");
          }
        }

        std::vector<double> extra_distances(extra_ids.size());
        if (!extra_ids.empty()) {
          GI_CUDA(cudaMemcpy(candidate_ids_d, extra_ids.data(),
                             extra_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
          const int candidate_blocks =
              (static_cast<int>(extra_ids.size()) + 255) / 256;
          exact_candidate_l2<<<candidate_blocks, 256>>>(
              full_pool_d, query, candidate_ids_d, candidate_distances_d,
              static_cast<int>(extra_ids.size()), static_cast<int>(header.dimension));
          GI_CUDA(cudaGetLastError());
          GI_CUDA(cudaDeviceSynchronize());
          GI_CUDA(cudaMemcpy(extra_distances.data(), candidate_distances_d,
                             extra_distances.size() * sizeof(double), cudaMemcpyDeviceToHost));
        }

        // A global top-k cannot contain a base object below the base-only k-th
        // rank, so the real GTS base top-k plus all visible sidecars and all
        // delta IDs is sufficient.  Re-evaluate only these selected base IDs
        // in host float64 for deterministic cross-tier ordering; no full base
        // or all-sidecar scan is performed here.
        std::vector<std::pair<double, int>> rows;
        rows.reserve(static_cast<std::size_t>(header.k) + extra_ids.size());
        for (std::size_t position = 0; position < gts_answer.topk.ids.size(); ++position) {
          const int base_id = gts_answer.topk.ids[position];
          if (base_id < 0 || base_id >= static_cast<int>(header.base_n) || state.active()[base_id] == 0) {
            die("GTS base top-k emitted an invalid/deleted base ID");
          }
          const float* vector = pool.at(base_id);
          double sum = 0.0;
          for (int d = 0; d < static_cast<int>(header.dimension); ++d) {
            const double diff = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
            sum += diff * diff;
          }
          rows.emplace_back(std::sqrt(sum), base_id);
        }
        for (std::size_t position = 0; position < extra_ids.size(); ++position) {
          rows.emplace_back(extra_distances[position], extra_ids[position]);
        }
        std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
          return left.first != right.first ? left.first < right.first : left.second < right.second;
        });
        if (rows.size() > header.k) rows.resize(header.k);
        if (rows.size() != header.k) die("active pool has fewer than requested top-k results");

        visited_leaf_total += gts_answer.visited_leaf_ids.size();
        sidecar_candidate_distances += sidecar_ids.size();
        delta_candidate_distances += delta_ids.size();
        if (!sidecar_ids.empty()) ++query_with_sidecar_candidates;
        emit_query(output, event, rows, gts_answer.visited_leaf_ids,
                   static_cast<int>(sidecar_ids.size()),
                   static_cast<int>(delta_ids.size()));
        ++knn;
      } else if (event.op == kRange) {
        die("G1 deliberately rejects range events: stable-ID dynamic range exporter is not implemented");
      } else {
        die("unknown trace op");
      }
      if (!output) die("failed to write result JSONL");
    }
    output.close();
    frozen.assert_unchanged(runtime);

    cudaDeviceProp property{};
    int device = 0;
    GI_CUDA(cudaGetDevice(&device));
    GI_CUDA(cudaGetDeviceProperties(&property, device));
    std::ofstream summary(args.summary);
    if (!summary) die("cannot write summary JSON");
    summary << "{\n"
            << "  \"schema\": \"safe-c1-g1-topk-v1\",\n"
            << "  \"status\": \"PASS_G1_PENDING_INDEPENDENT_VALIDATOR\",\n"
            << "  \"scope\": \"G1 correctness only: real GTS base traversal receipt + visited-leaf-only sidecar CUDA scan + exact global delta CUDA scan; no range, base delete, rebuild, or latency claim\",\n"
            << "  \"archived_incremental_updater_used\": false,\n"
            << "  \"base_tree\": {\"base_n\": " << header.base_n << ", \"dimension\": " << header.dimension
            << ", \"tree_height\": " << runtime.tree_height << ", \"frozen_hash\": " << frozen.hash
            << ", \"frozen_logical_leaf_id_hash\": " << frozen.logical_leaf_id_hash
            << ", \"frozen_logical_leaf_id_count\": " << frozen.logical_leaf_id_count
            << ", \"metric_encoding\": {\"infi_dis\": " << metric_encoding.infi_dis
            << ", \"dis_code\": " << metric_encoding.dis_code
            << ", \"bbox_squared_upper_bound\": " << metric_encoding.bbox_squared_upper_bound
            << ", \"bbox_l2_upper_bound\": " << json_float(static_cast<double>(metric_encoding.bbox_l2_upper_bound))
            << ", \"policy\": \"max(10000, ceil(pool_plus_query_bbox_l2_upper_bound)+1024)\""
            << ", \"raised_above_legacy_default\": " << (metric_encoding.raised_above_legacy_default ? "true" : "false")
            << "}},\n"
            << "  \"gts_static_vector_topk_probe\": {\"status\": \"PASS\", \"queries\": " << header.query_n
            << ", \"k\": " << header.k << ", \"id_mismatch\": " << gts_probe_id_mismatch
            << ", \"distance_mismatch\": " << gts_probe_distance_mismatch
            << ", \"ambiguous_boundary_ties\": " << gts_probe_ambiguous_boundary_ties
            << ", \"internal_tie_groups\": " << gts_probe_internal_tie_groups
            << ", \"positional_tie_reorders\": " << gts_probe_positional_tie_reorders
            << ", \"id_comparison\": \"exact per untied rank; equal-distance groups compare stable-ID set; k-boundary ties block proof\""
            << ", \"distance_comparator\": {\"absolute_floor\": " << kProbeDistanceAbsoluteFloor
            << ", \"max_float_ulps\": " << kProbeDistanceMaxFloatUlps
            << ", \"observed_max_abs_delta\": " << json_float(gts_probe_max_abs_distance_delta)
            << ", \"observed_max_relative_delta\": " << json_float(gts_probe_max_relative_distance_delta)
            << ", \"reference_arithmetic\": \"float64 L2 then float32; real GTS static path float32\"}},\n"
            << "  \"dynamic_export\": {\"method\": \"real GTS base top-k + visited-leaf-only sidecar scan + global delta scan\", \"knn\": " << knn
            << ", \"range\": 0, \"visited_leaf_total\": " << visited_leaf_total
            << ", \"query_with_sidecar_candidates\": " << query_with_sidecar_candidates
            << ", \"sidecar_candidate_distances\": " << sidecar_candidate_distances
            << ", \"delta_candidate_distances\": " << delta_candidate_distances << "},\n"
            << "  \"updates\": {\"insert\": " << inserts << ", \"delete\": " << deletes
            << ", \"direct_insert\": " << state.direct_inserts() << ", \"delta_insert\": " << state.delta_inserts()
            << ", \"certificate_reject\": " << state.certificate_rejects() << ", \"capacity_reject\": " << state.capacity_rejects()
            << ", \"direct_live_final\": " << state.direct_live() << ", \"delta_live_final\": " << state.delta_live() << "},\n"
            << "  \"final_active_count\": " << state.active_count() << ",\n"
            << "  \"gpu\": {\"name\": \"" << property.name << "\", \"compute_capability\": \""
            << property.major << '.' << property.minor << "\", \"sm_count\": " << property.multiProcessorCount << "},\n"
            << "  \"not_established\": [\"range-ID exporter\", \"base-delete handling\", \"rebuild\", \"selective dynamic latency\", \"performance benefit\"]\n"
            << "}\n";
    summary.close();

    GI_CUDA(cudaFree(full_pool_d)); full_pool_d = nullptr;
    GI_CUDA(cudaFree(candidate_ids_d)); candidate_ids_d = nullptr;
    GI_CUDA(cudaFree(candidate_distances_d)); candidate_distances_d = nullptr;
    GI_CUDA(cudaFree(query_d)); query_d = nullptr;
    release_tree(runtime);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    if (full_pool_d) cudaFree(full_pool_d);
    if (candidate_ids_d) cudaFree(candidate_ids_d);
    if (candidate_distances_d) cudaFree(candidate_distances_d);
    if (query_d) cudaFree(query_d);
    release_tree(runtime);
    return 2;
  }
}
