// Isolated Safe-C1 G1 CUDA/GTS top-k visibility gate.
//
// This file deliberately does NOT include incremental_insert.cuh or update.cuh.
// It implements real GTS base top-k traversal receipts plus visited-leaf sidecar
// and global-delta candidate merging. Range-ID export, base deletion, and rebuild
// remain explicitly out of scope; no legacy direct-insert path is evidence for E1-G.

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
#include "safe_c1_tie_free_witness.hpp"

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

  // `max_distance` remains fingerprinted in compute_hash() to detect frozen
  // tree mutation, but it is not a routing bound: search_v2's nodeProcessKnn
  // derives the upper boundary from the physical next sibling's min_dis.
  [[nodiscard]] bool valid_nonempty_node(int node_id) const {
    return node_id >= 0 && node_id < static_cast<int>(nodes.size()) &&
           node_id < static_cast<int>(empty.size()) && empty[node_id] == 0;
  }

  // Fail closed unless this parent has the physical full TREE_ORDER sibling
  // block that search_v2 can dereference, one common pivot, and strictly
  // increasing finite min_dis values. This is deliberately stronger than
  // merely finding one non-empty child: any partial/corrupt sibling block goes
  // to exact global delta rather than receiving a direct sidecar placement.
  [[nodiscard]] int common_pivot_or_reject(int parent) const {
    if (fanout != TREE_ORDER || parent < 0) return -1;
    int pivot = -1;
    float previous_min = 0.0F;
    for (int slot = 0; slot < fanout; ++slot) {
      const int child = parent * fanout + slot + 1;
      if (!valid_nonempty_node(child)) return -1;
      const TN& node = nodes[child];
      if (node.pid < 0 || !std::isfinite(node.min_dis)) return -1;
      if (pivot == -1) pivot = node.pid;
      if (node.pid != pivot) return -1;
      if (slot > 0 && !(node.min_dis > previous_min + kStrictEpsilon)) {
        return -1;
      }
      previous_min = node.min_dis;
    }
    return pivot;
  }

  // Exact host mirror of source_gts_incremental/include/search_v2.cuh:
  // non-last child i accepts min_i + eps < r < min_(i+1) - eps; the final
  // child has only r > min_last + eps. No max-distance upper bound is invented.
  [[nodiscard]] int only_strict_search_native_child(int parent, float distance) const {
    if (!std::isfinite(distance) || distance < 0.0F ||
        common_pivot_or_reject(parent) < 0) {
      return -1;
    }
    int match = -1;
    for (int slot = 0; slot < fanout; ++slot) {
      const int child = parent * fanout + slot + 1;
      // common_pivot_or_reject already audited every physical sibling, but keep
      // the local check so an accidental future refactor still fails closed.
      if (!valid_nonempty_node(child) || !std::isfinite(nodes[child].min_dis)) {
        return -1;
      }
      bool upper_ok = true;
      if (slot + 1 < fanout) {
        const int next = child + 1;
        if (!valid_nonempty_node(next) || !std::isfinite(nodes[next].min_dis)) {
          return -1;
        }
        upper_ok = distance < nodes[next].min_dis - kStrictEpsilon;
      }
      if (distance > nodes[child].min_dis + kStrictEpsilon && upper_ok) {
        if (match != -1) return -1;
        match = child;
      }
    }
    return match;
  }

  // Returns a leaf only when every visited ancestor has all ten physical
  // non-empty siblings, one common pivot, and strictly increasing sibling
  // minima under the search_v2 boundary contract. Otherwise return -1 and the
  // caller puts the object in exact global delta (fail closed).
  [[nodiscard]] int certify_leaf(const HostVectorPool& pool,
                                 StableId inserted) const {
    if (!initialized() || tree_height <= 1 || fanout != TREE_ORDER) return -1;
    int current = 0;
    for (int level = 0; level < tree_height - 1; ++level) {
      const int pivot = common_pivot_or_reject(current);
      if (pivot < 0) return -1;
      const int child = only_strict_search_native_child(
          current, pool.l2(inserted, pivot));
      if (child < 0) return -1;
      if (nodes[child].is_leaf == 1) return child;
      current = child;
    }
    return -1;
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
  // v3 is deliberately a single bounded G1B capacity witness runner, rather
  // than a general update benchmark. The contract pins the selected IDs and
  // exact nine-event trace before CUDA/tree construction.
  std::string capacity_witness_contract;
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
    else if (key == "--capacity-witness-contract") args.capacity_witness_contract = value();
    else if (key == "--leaf-capacity") args.leaf_capacity = std::stoi(value());
    else if (key == "--help") {
      std::cout << "usage: GTS_safe_c1_g1b_v3 --bundle QUANTIZED_BUNDLE --out RESULTS.jsonl "
                   "--summary SUMMARY.json --capacity-witness-contract CONTRACT.txt --leaf-capacity 2\n";
      std::exit(0);
    } else {
      die("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.output.empty() || args.summary.empty() ||
      args.capacity_witness_contract.empty()) {
    die("--bundle, --out, --summary, and --capacity-witness-contract are required");
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

// The bundle format declares mapping payloads as little-endian signed i32.
// This runner does not implement arbitrary stable-ID remapping: every GTS
// path addresses pool[stable_id], and the base is [0, base_n). Decode bytes
// explicitly rather than assuming host endianness, then fail closed unless the
// audit files certify that exact identity layout before any CUDA/tree action.
std::vector<std::uint32_t> read_little_endian_i32_identity_payload(
    const std::string& path, std::size_t count) {
  if (count > std::numeric_limits<std::size_t>::max() / 4U) {
    die("stable-ID mapping byte count overflow for " + path);
  }
  const std::size_t bytes = count * 4U;
  std::ifstream input(path, std::ios::binary);
  if (!input) die("cannot open stable-ID mapping " + path);
  std::vector<unsigned char> raw(bytes);
  input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
  if (input.gcount() != static_cast<std::streamsize>(bytes)) {
    die("unexpected size for stable-ID mapping " + path);
  }
  char extra = 0;
  if (input.read(&extra, 1)) die("extra bytes in stable-ID mapping " + path);
  std::vector<std::uint32_t> values(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t offset = index * 4U;
    values[index] = static_cast<std::uint32_t>(raw[offset]) |
                    (static_cast<std::uint32_t>(raw[offset + 1]) << 8U) |
                    (static_cast<std::uint32_t>(raw[offset + 2]) << 16U) |
                    (static_cast<std::uint32_t>(raw[offset + 3]) << 24U);
  }
  return values;
}

void require_identity_stable_id_layout(const std::string& bundle, const TraceHeader& header) {
  const auto stable_to_row = read_little_endian_i32_identity_payload(
      bundle + "/stable_id_to_pool_row.i32", static_cast<std::size_t>(header.pool_n));
  const auto initial_base_ids = read_little_endian_i32_identity_payload(
      bundle + "/initial_base_stable_ids.i32", static_cast<std::size_t>(header.base_n));
  for (std::size_t stable_id = 0; stable_id < stable_to_row.size(); ++stable_id) {
    if (stable_to_row[stable_id] != static_cast<std::uint32_t>(stable_id)) {
      die("stable_id_to_pool_row.i32 is not the required identity layout at stable ID " +
          std::to_string(stable_id));
    }
  }
  for (std::size_t base_id = 0; base_id < initial_base_ids.size(); ++base_id) {
    if (initial_base_ids[base_id] != static_cast<std::uint32_t>(base_id)) {
      die("initial_base_stable_ids.i32 is not the required identity base prefix at index " +
          std::to_string(base_id));
    }
  }
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

// The G1B evidence is intentionally not a general workload.  This compact,
// text-based contract is generated only from the externally frozen READY v2
// selection receipt and is parsed before the first CUDA operation.  It binds
// the exact capacity-two sequence and preserves the distinction between a
// certificate result and a capacity fallback.
struct G1BCapacityWitnessContract {
  int leaf_capacity = -1;
  int sidecar_leaf_id = -1;
  int a = -1;
  int b = -1;
  int c = -1;
  int query_a = -1;
  int query_b = -1;
  int query_c = -1;
  std::string v2_selection_sha256;
};

std::string trim_ascii(std::string value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1);
}

int parse_contract_int(const std::string& key, const std::string& text) {
  std::size_t parsed = 0;
  try {
    const int value = std::stoi(text, &parsed);
    if (parsed != text.size()) die("non-integer G1B contract field " + key);
    return value;
  } catch (const std::exception&) {
    die("invalid integer G1B contract field " + key);
  }
}

G1BCapacityWitnessContract read_g1b_capacity_witness_contract(
    const std::string& path) {
  std::ifstream input(path);
  if (!input) die("cannot open G1B capacity witness contract " + path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    line = trim_ascii(line);
    if (line.empty() || line[0] == '#') continue;
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || line.find('=', separator + 1) != std::string::npos) {
      die("malformed G1B capacity witness contract line");
    }
    const std::string key = trim_ascii(line.substr(0, separator));
    const std::string value = trim_ascii(line.substr(separator + 1));
    if (key.empty() || value.empty() || values.find(key) != values.end()) {
      die("empty or duplicate G1B capacity witness contract field");
    }
    values.emplace(key, value);
  }
  const auto field = [&](const std::string& key) -> const std::string& {
    const auto it = values.find(key);
    if (it == values.end()) die("missing G1B capacity witness contract field " + key);
    return it->second;
  };
  if (field("schema") != "safe-c1-g1b-capacity-witness-contract-v3") {
    die("wrong G1B capacity witness contract schema");
  }
  G1BCapacityWitnessContract contract;
  contract.leaf_capacity = parse_contract_int("leaf_capacity", field("leaf_capacity"));
  contract.sidecar_leaf_id = parse_contract_int("sidecar_leaf_id", field("sidecar_leaf_id"));
  contract.a = parse_contract_int("stable_id_a", field("stable_id_a"));
  contract.b = parse_contract_int("stable_id_b", field("stable_id_b"));
  contract.c = parse_contract_int("stable_id_c", field("stable_id_c"));
  contract.query_a = parse_contract_int("query_id_a", field("query_id_a"));
  contract.query_b = parse_contract_int("query_id_b", field("query_id_b"));
  contract.query_c = parse_contract_int("query_id_c", field("query_id_c"));
  contract.v2_selection_sha256 = field("v2_selection_sha256");
  if (contract.leaf_capacity != 2 || contract.sidecar_leaf_id < 0 ||
      contract.a < 0 || contract.b < 0 || contract.c < 0 ||
      contract.a == contract.b || contract.a == contract.c || contract.b == contract.c ||
      contract.query_a < 0 || contract.query_b < 0 || contract.query_c < 0 ||
      contract.v2_selection_sha256.size() != 64U) {
    die("invalid G1B capacity witness contract values");
  }
  static const std::vector<std::string> required = {
      "schema", "leaf_capacity", "sidecar_leaf_id", "stable_id_a", "stable_id_b",
      "stable_id_c", "query_id_a", "query_id_b", "query_id_c", "v2_selection_sha256"};
  if (values.size() != required.size()) die("unexpected G1B capacity witness contract field");
  return contract;
}

void validate_g1b_capacity_witness_trace(const TraceHeader& header,
                                         const std::vector<TraceEvent>& events,
                                         const G1BCapacityWitnessContract& contract,
                                         int requested_leaf_capacity) {
  if (requested_leaf_capacity != contract.leaf_capacity || requested_leaf_capacity != 2) {
    die("G1B capacity witness must run with leaf capacity exactly two");
  }
  if (header.query_n != 3 || header.event_count != 9 || events.size() != 9U) {
    die("G1B capacity witness must contain exactly three queries and nine events");
  }
  const std::vector<std::pair<int, int>> expected = {
      {kInsert, contract.a}, {kKnn, contract.query_a},
      {kInsert, contract.b}, {kKnn, contract.query_b},
      {kInsert, contract.c}, {kKnn, contract.query_c},
      {kDelete, contract.c}, {kDelete, contract.b}, {kDelete, contract.a}};
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (events[index].op_index != index || events[index].op != expected[index].first ||
        events[index].argument != expected[index].second) {
      die("trace does not match the strict G1B capacity-two sequence at op " +
          std::to_string(index));
    }
  }
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

enum class InsertFallbackReason { kNone, kCertificateReject, kCapacityFull };

const char* fallback_reason_label(InsertFallbackReason reason) {
  switch (reason) {
    case InsertFallbackReason::kNone: return "none";
    case InsertFallbackReason::kCertificateReject: return "certificate_reject";
    case InsertFallbackReason::kCapacityFull: return "capacity_full";
  }
  return "unknown";
}

// This receipt intentionally carries both the pure search-native certificate
// outcome and the final placement.  In particular, a capacity fallback keeps
// its certified leaf ID: otherwise a later report could conflate a failed
// certificate with a successful certificate that lost only because its sidecar
// was full.
struct InsertReceipt {
  safe_c1::PlacementKind placement = safe_c1::PlacementKind::kUnknown;
  int certified_leaf_id = -1;
  InsertFallbackReason fallback_reason = InsertFallbackReason::kNone;
};

class IntegrationState {
 public:
  IntegrationState(int pool_n, int base_n, int leaf_capacity)
      : active_(pool_n, 0), placement_(pool_n, safe_c1::PlacementKind::kDeleted),
        sidecar_leaf_for_(pool_n, -1), certified_leaf_for_(pool_n, -1),
        fallback_reason_for_(pool_n, InsertFallbackReason::kNone),
        leaf_capacity_(leaf_capacity) {
    if (pool_n <= 0 || base_n <= 0 || base_n > pool_n) die("bad state dimensions");
    for (int id = 0; id < base_n; ++id) {
      active_[id] = 1;
      placement_[id] = safe_c1::PlacementKind::kBase;
    }
  }

  InsertReceipt insert(int id, const safe_c1::FrozenTreeSnapshot& frozen,
                       const safe_c1::HostVectorPool& pool) {
    check(id);
    if (active_[id] != 0) die("insert targets active stable ID " + std::to_string(id));
    active_[id] = 1;
    const int certified_leaf = frozen.certify_leaf(pool, id);
    certified_leaf_for_[id] = certified_leaf;
    sidecar_leaf_for_[id] = -1;
    if (certified_leaf >= 0 &&
        static_cast<int>(sidecars_[certified_leaf].size()) < leaf_capacity_) {
      sidecars_[certified_leaf].push_back(id);
      placement_[id] = safe_c1::PlacementKind::kDirect;
      sidecar_leaf_for_[id] = certified_leaf;
      fallback_reason_for_[id] = InsertFallbackReason::kNone;
      ++direct_inserts_;
      return InsertReceipt{placement_[id], certified_leaf, fallback_reason_for_[id]};
    }
    placement_[id] = safe_c1::PlacementKind::kDelta;
    delta_.push_back(id);
    if (certified_leaf < 0) {
      fallback_reason_for_[id] = InsertFallbackReason::kCertificateReject;
      ++certificate_rejects_;
    } else {
      fallback_reason_for_[id] = InsertFallbackReason::kCapacityFull;
      ++capacity_rejects_;
    }
    ++delta_inserts_;
    return InsertReceipt{placement_[id], certified_leaf, fallback_reason_for_[id]};
  }

  void erase(int id) {
    check(id);
    if (active_[id] == 0) die("delete targets inactive stable ID " + std::to_string(id));
    const safe_c1::PlacementKind prior = placement_[id];
    if (prior == safe_c1::PlacementKind::kDirect) {
      const int leaf = sidecar_leaf_for_[id];
      auto it = sidecars_.find(leaf);
      if (it == sidecars_.end()) die("missing direct sidecar placement");
      auto& values = it->second;
      values.erase(std::remove(values.begin(), values.end(), id), values.end());
    } else if (prior == safe_c1::PlacementKind::kDelta) {
      delta_.erase(std::remove(delta_.begin(), delta_.end(), id), delta_.end());
    }
    active_[id] = 0;
    placement_[id] = safe_c1::PlacementKind::kDeleted;
    sidecar_leaf_for_[id] = -1;
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
  // Actual sidecar placement, never the certificate leaf for a delta fallback.
  int leaf_of(int id) const {
    check(id);
    return sidecar_leaf_for_[id];
  }
  int certified_leaf_of(int id) const {
    check(id);
    return certified_leaf_for_[id];
  }
  InsertFallbackReason fallback_reason_of(int id) const {
    check(id);
    return fallback_reason_for_[id];
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
  std::vector<int> sidecar_leaf_for_;
  std::vector<int> certified_leaf_for_;
  std::vector<InsertFallbackReason> fallback_reason_for_;
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

// The copied GTS base sort has no StableId tie-break.  This preflight is
// intentionally a witness-domain gate, not a repair for all tie-bearing inputs:
// only the static base top-(k+1) prefix must be tie-free under both exact and
// modeled GTS float keys (with matching first-k order), while every trace-state
// dynamic exact k-boundary must be distinct. Any violation throws ABORT_UNPROVED...
// before this runner can emit metadata or a summary PASS state.
safe_c1_tie::WitnessAudit validate_tie_free_witness_trace(
    const TraceHeader& header, const std::vector<TraceEvent>& events,
    const safe_c1_tie::TieFreeWitnessChecker& checker) {
  safe_c1_tie::WitnessAudit audit;
  checker.require_static_base_top_k_plus_one_tie_free(&audit);

  std::vector<std::uint8_t> active(header.pool_n, 0);
  std::fill(active.begin(), active.begin() + header.base_n, 1);
  for (const TraceEvent& event : events) {
    const int id = event.argument;
    if (event.op == kInsert) {
      if (id < static_cast<int>(header.base_n) || id >= static_cast<int>(header.pool_n)) {
        die("insert does not address disjoint reservoir at op " +
            std::to_string(event.op_index));
      }
      if (active[static_cast<std::size_t>(id)] != 0) {
        die("insert targets active stable ID " + std::to_string(id));
      }
      active[static_cast<std::size_t>(id)] = 1;
    } else if (event.op == kDelete) {
      if (id < 0 || id >= static_cast<int>(header.pool_n)) {
        die("delete stable ID out of range");
      }
      if (id < static_cast<int>(header.base_n)) {
        die("G1 refuses base deletion: compacting rebuild is an explicit G2 requirement");
      }
      if (active[static_cast<std::size_t>(id)] == 0) {
        die("delete targets inactive stable ID " + std::to_string(id));
      }
      active[static_cast<std::size_t>(id)] = 0;
    } else if (event.op == kKnn) {
      if (id < 0 || id >= static_cast<int>(header.query_n)) {
        die("query ID out of range");
      }
      checker.require_dynamic_k_boundary_tie_free(active, id, &audit);
    } else if (event.op == kRange) {
      die("G1 deliberately rejects range events: stable-ID dynamic range exporter is not implemented");
    } else {
      die("unknown trace op");
    }
  }
  return audit;
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
                 safe_c1::PlacementKind placement, int sidecar_leaf_id,
                 int certified_leaf_id, InsertFallbackReason fallback_reason) {
  output << "{\"record\":\"update\",\"op_index\":" << event.op_index
         << ",\"op\":\"" << (event.op == kInsert ? "insert" : "delete")
         << "\",\"stable_id\":" << event.argument
         << ",\"placement\":\"" << placement_label(placement) << "\""
         << ",\"sidecar_leaf_id\":" << sidecar_leaf_id
         << ",\"certified_leaf_id\":" << certified_leaf_id
         << ",\"fallback_reason\":\"" << fallback_reason_label(fallback_reason)
         << "\"}\n";
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
    const std::string expected_contract_path = args.bundle + "/g1b_capacity_contract.txt";
    if (args.capacity_witness_contract != expected_contract_path) {
      die("G1B capacity witness contract must be the bundle-local g1b_capacity_contract.txt");
    }
    const G1BCapacityWitnessContract capacity_contract =
        read_g1b_capacity_witness_contract(args.capacity_witness_contract);
    validate_g1b_capacity_witness_trace(header, events, capacity_contract,
                                        args.leaf_capacity);
    // Must precede every CUDA call and the GTS tree build: the executable itself
    // rejects any mapping that Python's exact active-set oracle would interpret
    // differently from the runner's stable-ID-as-pool-row semantics.
    require_identity_stable_id_layout(args.bundle, header);
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

    // Tie-free witness preflight runs before the first GTS kernel. It requires
    // a deterministic static top-(k+1) prefix under both exact and modeled GTS
    // float keys, plus matching first-k ID order; ties after that prefix are
    // allowed and remain outside any canonical (distance, stable_id) claim.
    const safe_c1_tie::TieFreeWitnessChecker tie_witness_checker(
        pool_i16, queries_i16, static_cast<int>(header.dimension),
        static_cast<int>(header.base_n), static_cast<int>(header.k));
    const safe_c1_tie::WitnessAudit tie_witness_audit =
        validate_tie_free_witness_trace(header, events, tie_witness_checker);

    // Configure only this isolated process before the first GTS kernel.  The
    // tree construction keeps its metric/routing semantics: within each pivot
    // block the positive normalization is order-preserving, while all encoded
    // distances remain strictly below one and therefore below DIS_CODE.
    const MetricEncodingPlan metric_encoding = make_metric_encoding_plan(
        pool_i16, queries_i16, static_cast<int>(header.dimension));
    DIS_CODE = metric_encoding.dis_code;
    INFI_DIS = metric_encoding.infi_dis;
    std::cout << "Safe-C1 G1B capacity witness metric_encoding infi_dis=" << INFI_DIS
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
    // CPU preflight already establishes a tie-free static top-(k+1) prefix
    // under both exact and modeled GTS float keys, with matching first-k order.
    // The probe still compares each equal-distance group as an ID set rather
    // than trusting an arbitrary positional order; it establishes no all-input
    // tie semantics.
    int gts_probe_same_distance_group_set_mismatch = 0;
    int gts_probe_invalid_or_duplicate_ids = 0;
    int gts_probe_distance_mismatch = 0;
    int gts_probe_internal_exact_tie_groups = 0;
    int gts_probe_internal_modeled_float_tie_groups = 0;
    int gts_probe_ambiguous_prefix_ties = 0;
    int gts_probe_exact_modeled_prefix_order_mismatch = 0;
    double gts_probe_max_abs_distance_delta = 0.0;
    double gts_probe_max_relative_distance_delta = 0.0;
    std::ostringstream gts_probe_distance_examples;
    std::ostringstream gts_probe_id_examples;
    int gts_probe_distance_examples_count = 0;
    int gts_probe_id_examples_count = 0;
    const int static_k = static_cast<int>(header.k);
    const int static_base_n = static_cast<int>(header.base_n);
    if (gts_probe.ids.size() != static_cast<std::size_t>(header.query_n) * header.k ||
        gts_probe.distances.size() != static_cast<std::size_t>(header.query_n) * header.k) {
      die("real GTS static base probe returned an unexpected result shape");
    }
    for (int qid = 0; qid < static_cast<int>(header.query_n); ++qid) {
      std::vector<std::pair<std::uint64_t, int>> exact_reference;
      std::vector<std::pair<float, int>> modeled_reference;
      exact_reference.reserve(static_cast<std::size_t>(static_base_n));
      modeled_reference.reserve(static_cast<std::size_t>(static_base_n));
      for (int stable_id = 0; stable_id < static_base_n; ++stable_id) {
        exact_reference.emplace_back(
            tie_witness_checker.exact_squared_l2_key_for(stable_id, qid), stable_id);
        modeled_reference.emplace_back(
            tie_witness_checker.modeled_gts_float_l2_key_for(stable_id, qid), stable_id);
      }
      std::sort(exact_reference.begin(), exact_reference.end());
      std::sort(modeled_reference.begin(), modeled_reference.end());
      const int static_prefix = std::min(static_base_n, static_k + 1);
      for (int rank = 1; rank < static_prefix; ++rank) {
        if (exact_reference[static_cast<std::size_t>(rank - 1)].first ==
            exact_reference[static_cast<std::size_t>(rank)].first ||
            modeled_reference[static_cast<std::size_t>(rank - 1)].first ==
            modeled_reference[static_cast<std::size_t>(rank)].first) {
          ++gts_probe_ambiguous_prefix_ties;
        }
      }
      for (int rank = 0; rank < static_k; ++rank) {
        if (exact_reference[static_cast<std::size_t>(rank)].second !=
            modeled_reference[static_cast<std::size_t>(rank)].second) {
          ++gts_probe_exact_modeled_prefix_order_mismatch;
        }
      }

      std::vector<std::pair<std::uint64_t, int>> expected_top_exact(
          exact_reference.begin(), exact_reference.begin() + static_k);
      std::vector<std::pair<std::uint64_t, int>> observed_top_exact;
      observed_top_exact.reserve(static_cast<std::size_t>(static_k));
      for (int position = 0; position < static_k; ++position) {
        const std::size_t offset = static_cast<std::size_t>(qid) * header.k + position;
        const int observed_id = gts_probe.ids[offset];
        if (observed_id < 0 || observed_id >= static_base_n) {
          ++gts_probe_invalid_or_duplicate_ids;
          if (gts_probe_id_examples_count < 5) {
            if (gts_probe_id_examples_count != 0) gts_probe_id_examples << ';';
            gts_probe_id_examples << "q=" << qid << ",p=" << position
                                  << ",invalid_id=" << observed_id;
            ++gts_probe_id_examples_count;
          }
          continue;
        }
        observed_top_exact.emplace_back(
            tie_witness_checker.exact_squared_l2_key_for(observed_id, qid), observed_id);
        const float observed = gts_probe.distances[offset];
        const float reference =
            tie_witness_checker.modeled_gts_float_l2_key_for(observed_id, qid);
        const double absolute_delta =
            std::fabs(static_cast<double>(observed) - static_cast<double>(reference));
        const double relative_delta =
            absolute_delta / std::max(1.0, std::fabs(static_cast<double>(reference)));
        gts_probe_max_abs_distance_delta =
            std::max(gts_probe_max_abs_distance_delta, absolute_delta);
        gts_probe_max_relative_distance_delta =
            std::max(gts_probe_max_relative_distance_delta, relative_delta);
        if (!close_distance(observed, reference)) {
          ++gts_probe_distance_mismatch;
          if (gts_probe_distance_examples_count < 5) {
            if (gts_probe_distance_examples_count != 0) gts_probe_distance_examples << ';';
            gts_probe_distance_examples << "q=" << qid << ",p=" << position
                                        << ",id=" << observed_id
                                        << ",obs=" << std::setprecision(9) << observed
                                        << ",modeled_ref=" << reference
                                        << ",abs=" << absolute_delta
                                        << ",rel=" << relative_delta;
            ++gts_probe_distance_examples_count;
          }
        }
      }
      std::sort(observed_top_exact.begin(), observed_top_exact.end());
      // This is intentionally a same-distance-group set comparison: sorting by
      // (exact squared-L2, StableId) canonicalizes IDs within each equal-distance
      // group without imposing the GTS output's positional order.
      if (observed_top_exact != expected_top_exact) {
        ++gts_probe_same_distance_group_set_mismatch;
        if (gts_probe_id_examples_count < 5) {
          if (gts_probe_id_examples_count != 0) gts_probe_id_examples << ';';
          gts_probe_id_examples << "q=" << qid << ",expected_exact_groups=";
          for (std::size_t index = 0; index < expected_top_exact.size(); ++index) {
            if (index) gts_probe_id_examples << ',';
            gts_probe_id_examples << expected_top_exact[index].second;
          }
          gts_probe_id_examples << ",observed_exact_groups=";
          for (std::size_t index = 0; index < observed_top_exact.size(); ++index) {
            if (index) gts_probe_id_examples << ',';
            gts_probe_id_examples << observed_top_exact[index].second;
          }
          ++gts_probe_id_examples_count;
        }
      }
      for (int begin = 0; begin < static_k;) {
        int end = begin + 1;
        while (end < static_k &&
               expected_top_exact[static_cast<std::size_t>(end)].first ==
               expected_top_exact[static_cast<std::size_t>(begin)].first) {
          ++end;
        }
        if (end - begin > 1) ++gts_probe_internal_exact_tie_groups;
        begin = end;
      }
      for (int begin = 0; begin < static_k;) {
        int end = begin + 1;
        while (end < static_k &&
               modeled_reference[static_cast<std::size_t>(end)].first ==
               modeled_reference[static_cast<std::size_t>(begin)].first) {
          ++end;
        }
        if (end - begin > 1) ++gts_probe_internal_modeled_float_tie_groups;
        begin = end;
      }
    }
    if (gts_probe_same_distance_group_set_mismatch != 0 ||
        gts_probe_invalid_or_duplicate_ids != 0 || gts_probe_distance_mismatch != 0 ||
        gts_probe_ambiguous_prefix_ties != 0 ||
        gts_probe_exact_modeled_prefix_order_mismatch != 0) {
      std::ostringstream detail;
      detail << "real GTS static base probe mismatch: same_distance_group_set="
             << gts_probe_same_distance_group_set_mismatch
             << " invalid_or_duplicate_ids=" << gts_probe_invalid_or_duplicate_ids
             << " distance=" << gts_probe_distance_mismatch
             << " ambiguous_prefix_ties=" << gts_probe_ambiguous_prefix_ties
             << " exact_modeled_prefix_order=" << gts_probe_exact_modeled_prefix_order_mismatch
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
    output << "{\"record\":\"meta\",\"schema\":\"safe-c1-g1b-capacity-witness-v3-search-native\","
           << "\"scope\":\"G1B capacity-two witness only: strict I(a),Q(a),I(b),Q(b),I(c),Q(c),D(c),D(b),D(a); real GTS traversal receipt; visited-leaf-only sidecars; exact global delta; no timing, rebuild, range, base-delete, or complete-C3 claim\","
           << "\"capacity_witness_contract\":{\"leaf_capacity\":" << capacity_contract.leaf_capacity
           << ",\"sidecar_leaf_id\":" << capacity_contract.sidecar_leaf_id
           << ",\"stable_ids\":[" << capacity_contract.a << ',' << capacity_contract.b << ',' << capacity_contract.c << ']'
           << ",\"query_ids\":[" << capacity_contract.query_a << ',' << capacity_contract.query_b << ',' << capacity_contract.query_c << ']'
           << ",\"v2_selection_sha256\":\"" << capacity_contract.v2_selection_sha256 << "\"},"
           << "\"stable_id_layout\":{\"mode\":\"explicit_identity_only_current_v3_runner\",\"validated\":true},"
           << "\"gts_static_probe\":\"BOUNDED_TIE_WITNESS_VALIDATED\","
           << "\"tie_semantics\":{\"domain\":\"bounded static-prefix and dynamic-boundary witness only\","
           << "\"static_base_oracle_policy\":\"first min(k+1,base_n) exact squared-L2 and modeled GTS float-L2 keys pairwise distinct; first-k ID order agrees\","
           << "\"dynamic_k_boundary_policy\":\"k and k+1 exact quantized squared-L2 keys distinct for every trace kNN state\","
           << "\"static_base_queries_checked\":" << tie_witness_audit.static_base_queries_checked
           << ",\"dynamic_k_boundary_queries_checked\":" << tie_witness_audit.dynamic_k_boundary_queries_checked
           << ",\"status\":\"BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED\","
           << "\"on_violation\":\"ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN\","
           << "\"global_canonical_distance_stable_id_proof\":false}}\n";

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
        const InsertReceipt receipt = state.insert(argument, frozen, pool);
        frozen.assert_unchanged(runtime);
        if (event.op_index == 0 || event.op_index == 2) {
          if (receipt.placement != safe_c1::PlacementKind::kDirect ||
              receipt.certified_leaf_id != capacity_contract.sidecar_leaf_id ||
              receipt.fallback_reason != InsertFallbackReason::kNone ||
              state.leaf_of(argument) != capacity_contract.sidecar_leaf_id) {
            die("G1B requires a/b direct placement at the externally selected same leaf");
          }
        } else if (event.op_index == 4) {
          if (receipt.placement != safe_c1::PlacementKind::kDelta ||
              receipt.certified_leaf_id != capacity_contract.sidecar_leaf_id ||
              receipt.fallback_reason != InsertFallbackReason::kCapacityFull ||
              state.leaf_of(argument) != -1 || state.capacity_rejects() != 1 ||
              state.certificate_rejects() != 0) {
            die("G1B requires c to pass the same-leaf certificate then fall back only for capacity");
          }
        } else {
          die("unexpected insert position after strict G1B trace validation");
        }
        emit_update(output, event, receipt.placement, state.leaf_of(argument),
                    receipt.certified_leaf_id, receipt.fallback_reason);
        ++inserts;
      } else if (event.op == kDelete) {
        if (argument < 0 || argument >= static_cast<int>(header.pool_n)) {
          die("delete stable ID out of range");
        }
        const safe_c1::PlacementKind prior = state.placement_of(argument);
        if (prior == safe_c1::PlacementKind::kBase) {
          die("G1 refuses base deletion: compacting rebuild is an explicit G2 requirement");
        }
        const int prior_sidecar_leaf = state.leaf_of(argument);
        const int prior_certified_leaf = state.certified_leaf_of(argument);
        const InsertFallbackReason prior_fallback = state.fallback_reason_of(argument);
        if (event.op_index == 6) {
          if (prior != safe_c1::PlacementKind::kDelta || prior_sidecar_leaf != -1 ||
              prior_certified_leaf != capacity_contract.sidecar_leaf_id ||
              prior_fallback != InsertFallbackReason::kCapacityFull) {
            die("G1B delete(c) no longer records the required capacity fallback receipt");
          }
        } else if (event.op_index == 7 || event.op_index == 8) {
          if (prior != safe_c1::PlacementKind::kDirect ||
              prior_sidecar_leaf != capacity_contract.sidecar_leaf_id ||
              prior_certified_leaf != capacity_contract.sidecar_leaf_id ||
              prior_fallback != InsertFallbackReason::kNone) {
            die("G1B delete(a/b) no longer records the required direct same-leaf receipt");
          }
        } else {
          die("unexpected delete position after strict G1B trace validation");
        }
        state.erase(argument);
        frozen.assert_unchanged(runtime);
        emit_update(output, event, safe_c1::PlacementKind::kDeleted, prior_sidecar_leaf,
                    prior_certified_leaf, prior_fallback);
        ++deletes;
      } else if (event.op == kKnn) {
        if (argument < 0 || argument >= static_cast<int>(header.query_n)) {
          die("query ID out of range");
        }
        float* query = query_d + static_cast<std::size_t>(argument) * header.dimension;

        // Re-check the exact active-set k boundary immediately before the
        // GTS call. The whole trace was preflighted above; this defensive gate
        // ensures any future state-transition divergence aborts unproved rather
        // than emitting a successful tie-bearing witness.
        tie_witness_checker.require_dynamic_k_boundary_tie_free(
            state.active(), argument, nullptr);

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
        const std::vector<int> expected_sidecar_ids =
            event.op_index == 1 ? std::vector<int>{capacity_contract.a} :
            std::vector<int>{capacity_contract.a, capacity_contract.b};
        const std::vector<int> expected_delta_ids =
            event.op_index == 5 ? std::vector<int>{capacity_contract.c} : std::vector<int>{};
        if (sidecar_ids != expected_sidecar_ids || delta_ids != expected_delta_ids ||
            std::find(gts_answer.visited_leaf_ids.begin(), gts_answer.visited_leaf_ids.end(),
                      capacity_contract.sidecar_leaf_id) == gts_answer.visited_leaf_ids.end()) {
          die("G1B query does not expose exactly the expected real-traversal sidecar/delta tiers");
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
        const int expected_top1 = event.op_index == 1 ? capacity_contract.a :
                                  event.op_index == 3 ? capacity_contract.b : capacity_contract.c;
        if (rows.empty() || rows.front().second != expected_top1) {
          die("G1B self-query witness did not return the inserted object as top-1");
        }

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
    if (inserts != 3 || deletes != 3 || knn != 3 || state.direct_inserts() != 2 ||
        state.delta_inserts() != 1 || state.certificate_rejects() != 0 ||
        state.capacity_rejects() != 1 || state.direct_live() != 0 ||
        state.delta_live() != 0 || state.active_count() != static_cast<int>(header.base_n)) {
      die("G1B terminal state does not match the strict capacity-two witness contract");
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
            << "  \"schema\": \"safe-c1-g1b-capacity-witness-v3-search-native\",\n"
            << "  \"status\": \"PASS_G1B_CAPACITY_WITNESS_PENDING_INDEPENDENT_VALIDATOR\",\n"
            << "  \"scope\": \"G1B capacity-two witness only: strict three-object same-leaf sequence; real GTS traversal receipt + visited-leaf-only sidecar CUDA scan + exact global delta CUDA scan; no range, base delete, rebuild, latency, performance, or all-input tie claim\",\n"
            << "  \"stable_id_layout\": {\"mode\": \"explicit_identity_only_current_v3_runner\", \"validated\": true},\n"
            << "  \"archived_incremental_updater_used\": false,\n"
            << "  \"capacity_witness_contract\": {\"leaf_capacity\": " << capacity_contract.leaf_capacity
            << ", \"sidecar_leaf_id\": " << capacity_contract.sidecar_leaf_id
            << ", \"stable_ids\": [" << capacity_contract.a << ',' << capacity_contract.b << ',' << capacity_contract.c << ']'
            << ", \"query_ids\": [" << capacity_contract.query_a << ',' << capacity_contract.query_b << ',' << capacity_contract.query_c << ']'
            << ", \"v2_selection_sha256\": \"" << capacity_contract.v2_selection_sha256 << "\"},\n"
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
            << ", \"k\": " << header.k
            << ", \"same_distance_group_set_mismatch\": " << gts_probe_same_distance_group_set_mismatch
            << ", \"invalid_or_duplicate_ids\": " << gts_probe_invalid_or_duplicate_ids
            << ", \"distance_mismatch\": " << gts_probe_distance_mismatch
            << ", \"ambiguous_prefix_ties\": " << gts_probe_ambiguous_prefix_ties
            << ", \"internal_exact_tie_groups\": " << gts_probe_internal_exact_tie_groups
            << ", \"internal_modeled_gts_float_tie_groups\": " << gts_probe_internal_modeled_float_tie_groups
            << ", \"exact_modeled_prefix_order_mismatch\": " << gts_probe_exact_modeled_prefix_order_mismatch
            << ", \"id_comparison\": \"same-distance-group ID-set comparison against exact squared-L2 top-k; no all-input tie proof\""
            << ", \"distance_comparator\": {\"absolute_floor\": " << kProbeDistanceAbsoluteFloor
            << ", \"max_float_ulps\": " << kProbeDistanceMaxFloatUlps
            << ", \"observed_max_abs_delta\": " << json_float(gts_probe_max_abs_distance_delta)
            << ", \"observed_max_relative_delta\": " << json_float(gts_probe_max_relative_distance_delta)
            << ", \"reference_arithmetic\": \"modeled GTS float diff/product + double sum + sqrtf(float(sum))\"}},\n"
            << "  \"tie_semantics\": {\"domain\": \"bounded static-prefix and dynamic-boundary witness only\", \"status\": \"BOUNDED_TIE_WITNESS_PRECONDITION_SATISFIED\", \"static_base_queries_checked\": " << tie_witness_audit.static_base_queries_checked
            << ", \"dynamic_k_boundary_queries_checked\": " << tie_witness_audit.dynamic_k_boundary_queries_checked
            << ", \"static_base_oracle_policy\": \"first min(k+1,base_n) exact squared-L2 and modeled GTS float-L2 keys pairwise distinct; first-k ID order agrees\", \"dynamic_k_boundary_policy\": \"k and k+1 exact quantized squared-L2 keys distinct for every trace kNN state\", \"on_violation\": \"ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN\", \"global_canonical_distance_stable_id_proof\": false},\n"
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
            << "  \"not_established\": [\"range-ID exporter\", \"base-delete handling\", \"rebuild\", \"selective dynamic latency\", \"performance benefit\", \"all-input tie semantics / global canonical (distance,stable_id) proof\"]\n"
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
