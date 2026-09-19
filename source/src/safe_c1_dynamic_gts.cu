// Isolated Safe-C1 G2 stable-ID seeded immediate-rebuild witness (v4).
//
// This file deliberately does NOT include incremental_insert.cuh or update.cuh.
// It keeps a full immutable StableId-addressed pool, builds each epoch from an
// explicit sorted live-ID seed BEFORE the first pivot pass, never writes legacy
// TN/id_list between rebuilds, and releases only obsolete tree metadata at the
// explicit base-delete -> immediate-rebuild transition.  It is a strict
// correctness witness, not a performance, general-rebuild, range, C2, or
// all-input tie-semantics claim.
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
#include <type_traits>

#define RP_DEFINE_CONSTANTS
#include "residual_pruning.cuh"
#include "tree.cuh"
#include "safe_search_v2.cuh"
#include "safe_c1_tie_free_witness.hpp"

// The archive config intentionally macro-rebinds `short` to float for this
// GTS build.  Give that physical encoding one explicit name: every full-pool
// allocation, device copy, and immutable-pool hash below uses this exact scalar
// type/byte count, never an unrelated host element type.
using GtsScalar = short;
static_assert(std::is_same<GtsScalar, float>::value,
              "this isolated runner requires the archive float GTS scalar encoding");
static_assert(sizeof(GtsScalar) == 4 && std::numeric_limits<float>::is_iec559,
              "G2 immutable-pool hash contract requires IEEE-754 float32 GtsScalar bytes");

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
  GtsScalar* data_d = nullptr;
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
// G2 seeded stable-ID rebuild witness executor (isolated v4)
// -----------------------------------------------------------------------------
// No archive mutable updater is included or called.  A rebuild is an explicit
// epoch transition: current active stable IDs seed a fresh GTS tree before its
// first pivot pass; sidecar/delta tiers are then cleared only after the new
// tree, exact leaf-ID set, pivot-ID subset, and static GTS probe have passed.

__global__ void initIndexDataSeededStableIds(
    int* data_info, TN* node_list, int* split_list, int* empty_list,
    int* id_list, const int* seeded_stable_ids) {
  const int id = blockDim.x * blockIdx.x + threadIdx.x;
  const int total = gridDim.x * blockDim.x;
  for (int idx = id; idx < data_info[1]; idx += total) {
    // This is the only semantic difference from archive tree.cuh:initIndexData:
    // the prevalidated live stable-ID seed is written BEFORE getPivotDis.
    id_list[idx] = seeded_stable_ids[idx];
  }
  if (id == 0) {
    node_list[0].size = data_info[1];
    node_list[0].lid = 0;
    node_list[0].pid = -1;
    node_list[0].min_dis = 0.0F;
    node_list[0].is_leaf = 0;
    split_list[0] = 1;
    empty_list[0] = 0;
  }
}

// Isolated copy of the archive constructor.  It preserves archive split/search
// mechanics and only replaces the contiguous-ID initialization kernel above.
// data_d intentionally owns the full immutable pool; data_info[1] is just the
// logical epoch base size and each seed value remains a valid data_d row.
void indexConstruSeededStableIds(GtsScalar* data_d, char* data_s, int* size_s,
                                 int* data_info, const std::vector<int>& seeded_stable_ids,
                                 int*& id_list, TN*& node_list, int*& max_node_num,
                                 int& tree_h, int*& empty_list) {
  if (data_d == nullptr || data_info == nullptr || data_info[0] <= 0 || data_info[1] <= 0 ||
      static_cast<int>(seeded_stable_ids.size()) != data_info[1]) {
    throw std::runtime_error("G2 seeded constructor: invalid full-pool data or seed cardinality");
  }
  if (!std::is_sorted(seeded_stable_ids.begin(), seeded_stable_ids.end()) ||
      std::adjacent_find(seeded_stable_ids.begin(), seeded_stable_ids.end()) != seeded_stable_ids.end() ||
      seeded_stable_ids.front() < 0) {
    throw std::runtime_error("G2 seeded constructor: seed stable IDs must be sorted nonnegative unique");
  }
  if (id_list != nullptr || node_list != nullptr || max_node_num != nullptr || empty_list != nullptr ||
      max_dis_d != nullptr || dis_list != nullptr || split_list != nullptr || split_num != nullptr ||
      pid_list != nullptr) {
    throw std::runtime_error("G2 seeded constructor: stale tree allocation/global detected before rebuild");
  }
  auto compute_max_height = [](int order, int object_count) {
    if (order <= 1 || object_count <= 0) return 1;
    int h = 1;
    long long leaves = 1;
    while (leaves * MAX_SIZE < object_count && h < 20) {
      leaves *= order;
      ++h;
    }
    return h;
  };

  MAX_H = compute_max_height(TREE_ORDER, data_info[1]);
  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&max_node_num), sizeof(int)));
  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&split_num), sizeof(int)));
  max_node_num[0] = static_cast<int>((std::pow(TREE_ORDER, MAX_H) - 1) / (TREE_ORDER - 1));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&split_list), max_node_num[0] * sizeof(int)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&pid_list), max_node_num[0] * sizeof(int)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&dis_list), data_info[1] * sizeof(double)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&empty_list), max_node_num[0] * sizeof(int)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&id_list), data_info[1] * sizeof(int)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&node_list), max_node_num[0] * sizeof(TN)));
  CHECK(cudaMalloc(reinterpret_cast<void**>(&max_dis_d), max_node_num[0] * sizeof(float)));
  CHECK(cudaMemset(max_dis_d, 0, max_node_num[0] * sizeof(float)));
  CHECK(cudaMemset(split_list, 0, max_node_num[0] * sizeof(int)));
  CHECK(cudaMemset(empty_list, 1, max_node_num[0] * sizeof(int)));
  split_num[0] = 1;
  cur_level = 0;
  start_idx = 0;

  int* seeded_ids_d = nullptr;
  CHECK(cudaMalloc(reinterpret_cast<void**>(&seeded_ids_d), data_info[1] * sizeof(int)));
  CHECK(cudaMemcpy(seeded_ids_d, seeded_stable_ids.data(), data_info[1] * sizeof(int),
                   cudaMemcpyHostToDevice));
  initIndexDataSeededStableIds<<<(data_info[1] - 1) / THREAD_NUM + 1, THREAD_NUM>>>(
      data_info, node_list, split_list, empty_list, id_list, seeded_ids_d);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  CHECK(cudaFree(seeded_ids_d));

  while (cur_level < MAX_H - 1 && split_num[0] > 0) {
    const int block_num = static_cast<int>(std::pow(TREE_ORDER, cur_level));
    getPivotDis<<<block_num, THREAD_NUM>>>(data_d, data_s, size_s, node_list, split_list,
                                            dis_list, id_list, start_idx, data_info, pid_list);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    thrust::sort_by_key(thrust::device, dis_list, dis_list + data_info[1], id_list);
    nodeSplit<<<block_num, THREAD_NUM>>>(node_list, split_list, dis_list, empty_list, start_idx,
                                          pid_list, data_d, id_list, data_info, data_s, size_s,
                                          max_dis_d);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    start_idx += static_cast<int>(std::pow(TREE_ORDER, cur_level));
    ++cur_level;
    split_num[0] = thrust::reduce(thrust::device, split_list, split_list + max_node_num[0], 0);
  }

  // Same leaf finalization/padding logic as the archived constructor.  Direct
  // Safe-C1 sidecars never write into this padded legacy id_list.
  {
    TN* h_nodes = static_cast<TN*>(std::malloc(max_node_num[0] * sizeof(TN)));
    int* h_empty = static_cast<int*>(std::malloc(max_node_num[0] * sizeof(int)));
    int* h_ids = static_cast<int*>(std::malloc(data_info[1] * sizeof(int)));
    if (h_nodes == nullptr || h_empty == nullptr || h_ids == nullptr) {
      std::free(h_nodes); std::free(h_empty); std::free(h_ids);
      throw std::runtime_error("G2 seeded constructor: host leaf workspace allocation failed");
    }
    CHECK(cudaMemcpy(h_nodes, node_list, max_node_num[0] * sizeof(TN), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_empty, empty_list, max_node_num[0] * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_ids, id_list, data_info[1] * sizeof(int), cudaMemcpyDeviceToHost));
    int fixed = 0;
    for (int i = 0; i < max_node_num[0]; ++i) {
      if (h_empty[i] == 0 && h_nodes[i].is_leaf == 0) {
        bool has_children = false;
        for (int slot = 0; slot < TREE_ORDER; ++slot) {
          const int child = i * TREE_ORDER + slot + 1;
          if (child < max_node_num[0] && h_empty[child] == 0) { has_children = true; break; }
        }
        if (!has_children) { h_nodes[i].is_leaf = 1; ++fixed; }
      }
    }
    if (fixed > 0) CHECK(cudaMemcpy(node_list, h_nodes, max_node_num[0] * sizeof(TN), cudaMemcpyHostToDevice));

    int leaf_count = 0;
    for (int i = 0; i < max_node_num[0]; ++i)
      if (h_empty[i] == 0 && h_nodes[i].is_leaf == 1) ++leaf_count;
    if (leaf_count <= 0) {
      std::free(h_nodes); std::free(h_empty); std::free(h_ids);
      throw std::runtime_error("G2 seeded constructor: no leaves after construction");
    }
    std::vector<int> leaf_ids;
    leaf_ids.reserve(static_cast<std::size_t>(leaf_count));
    for (int i = 0; i < max_node_num[0]; ++i)
      if (h_empty[i] == 0 && h_nodes[i].is_leaf == 1) leaf_ids.push_back(i);
    std::sort(leaf_ids.begin(), leaf_ids.end(), [&](int left, int right) {
      return h_nodes[left].lid < h_nodes[right].lid;
    });
    const int padded_total = data_info[1] + leaf_count * LEAF_PAD_SLOTS;
    int* h_padded = static_cast<int*>(std::malloc(padded_total * sizeof(int)));
    if (h_padded == nullptr) {
      std::free(h_nodes); std::free(h_empty); std::free(h_ids);
      throw std::runtime_error("G2 seeded constructor: padded ID workspace allocation failed");
    }
    std::memset(h_padded, -1, padded_total * sizeof(int));
    int write_pos = 0;
    for (int leaf_id : leaf_ids) {
      const int old_lid = h_nodes[leaf_id].lid;
      const int size = h_nodes[leaf_id].size;
      if (old_lid < 0 || size < 0 || old_lid + size > data_info[1]) {
        std::free(h_padded); std::free(h_nodes); std::free(h_empty); std::free(h_ids);
        throw std::runtime_error("G2 seeded constructor: invalid logical leaf range");
      }
      std::memcpy(h_padded + write_pos, h_ids + old_lid, size * sizeof(int));
      h_nodes[leaf_id].lid = write_pos;
      write_pos += size + LEAF_PAD_SLOTS;
    }
    CHECK(cudaFree(id_list));
    id_list = nullptr;
    CHECK(cudaMalloc(reinterpret_cast<void**>(&id_list), padded_total * sizeof(int)));
    CHECK(cudaMemcpy(id_list, h_padded, padded_total * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(node_list, h_nodes, max_node_num[0] * sizeof(TN), cudaMemcpyHostToDevice));
    std::free(h_padded); std::free(h_nodes); std::free(h_empty); std::free(h_ids);
  }

  tree_h = cur_level + 1;
  CHECK(cudaFree(dis_list)); dis_list = nullptr;
  CHECK(cudaFree(split_list)); split_list = nullptr;
  CHECK(cudaFree(split_num)); split_num = nullptr;
  CHECK(cudaFree(pid_list)); pid_list = nullptr;
}

namespace {

constexpr char kTraceMagic[8] = {'E', '1', 'G', 'T', 'R', 'C', '0', '2'};
constexpr std::uint32_t kTraceVersion = 2;
constexpr float kProbeDistanceAbsoluteFloor = 2.0e-3F;
constexpr int kProbeDistanceMaxFloatUlps = 2;

enum TraceOp : std::uint8_t { kInsert = 1, kDelete = 2, kKnn = 3, kRange = 4, kRebuild = 5 };

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
static_assert(sizeof(TraceHeader) == 48, "unexpected G2 trace header ABI");
static_assert(sizeof(TraceEvent) == 12, "unexpected G2 trace event ABI");

struct Args {
  std::string bundle;
  std::string output;
  std::string summary;
  std::string rebuild_contract;
  int leaf_capacity = -1;
};

[[noreturn]] void die(const std::string& message) {
  throw std::runtime_error("Safe-C1 G2 v4: " + message);
}

void cuda_or_die(cudaError_t status, const char* expression, const char* file, int line) {
  if (status == cudaSuccess) return;
  std::ostringstream out;
  out << "CUDA " << expression << " failed at " << file << ':' << line << ": "
      << cudaGetErrorString(status);
  die(out.str());
}
#define G2_CUDA(expr) cuda_or_die((expr), #expr, __FILE__, __LINE__)

Args parse_args(int argc, char** argv) {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string key(argv[index]);
    auto value = [&]() -> std::string {
      if (++index >= argc) die("missing value after " + key);
      return argv[index];
    };
    if (key == "--bundle") args.bundle = value();
    else if (key == "--out") args.output = value();
    else if (key == "--summary") args.summary = value();
    else if (key == "--g2-rebuild-contract") args.rebuild_contract = value();
    else if (key == "--leaf-capacity") args.leaf_capacity = std::stoi(value());
    else if (key == "--help") {
      std::cout << "usage: GTS_safe_c1_g2_v4 --bundle BUNDLE --out RESULTS.jsonl --summary SUMMARY.json "
                   "--g2-rebuild-contract CONTRACT.txt --leaf-capacity 1\n";
      std::exit(0);
    } else {
      die("unknown argument " + key);
    }
  }
  if (args.bundle.empty() || args.output.empty() || args.summary.empty() ||
      args.rebuild_contract.empty() || args.leaf_capacity <= 0) {
    die("--bundle, --out, --summary, --g2-rebuild-contract, and positive --leaf-capacity are required");
  }
  return args;
}

template <class T>
std::vector<T> read_exact_binary(const std::string& path, std::size_t count) {
  std::ifstream input(path, std::ios::binary);
  if (!input) die("cannot open " + path);
  std::vector<T> output(count);
  input.read(reinterpret_cast<char*>(output.data()), static_cast<std::streamsize>(count * sizeof(T)));
  if (input.gcount() != static_cast<std::streamsize>(count * sizeof(T))) die("unexpected size for " + path);
  char extra = 0;
  if (input.read(&extra, 1)) die("extra bytes in " + path);
  return output;
}

std::vector<int> read_i32_le(const std::string& path, std::size_t count) {
  const std::vector<unsigned char> raw = read_exact_binary<unsigned char>(path, count * 4U);
  std::vector<int> output(count);
  for (std::size_t index = 0; index < count; ++index) {
    const std::size_t offset = index * 4U;
    const std::uint32_t value = static_cast<std::uint32_t>(raw[offset]) |
        (static_cast<std::uint32_t>(raw[offset + 1]) << 8U) |
        (static_cast<std::uint32_t>(raw[offset + 2]) << 16U) |
        (static_cast<std::uint32_t>(raw[offset + 3]) << 24U);
    output[index] = static_cast<std::int32_t>(value);
  }
  return output;
}

std::vector<int> require_identity_layout(const std::string& bundle, const TraceHeader& header) {
  const std::vector<int> mapping = read_i32_le(bundle + "/stable_id_to_pool_row.i32", header.pool_n);
  const std::vector<int> initial = read_i32_le(bundle + "/initial_base_stable_ids.i32", header.base_n);
  for (int id = 0; id < static_cast<int>(mapping.size()); ++id) {
    if (mapping[static_cast<std::size_t>(id)] != id) die("stable_id_to_pool_row is not identity");
  }
  for (int id = 0; id < static_cast<int>(initial.size()); ++id) {
    if (initial[static_cast<std::size_t>(id)] != id) die("initial base is not identity prefix");
  }
  return initial;
}

std::pair<TraceHeader, std::vector<TraceEvent>> read_trace(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) die("cannot open trace " + path);
  TraceHeader header{};
  input.read(reinterpret_cast<char*>(&header), sizeof(header));
  if (input.gcount() != static_cast<std::streamsize>(sizeof(header))) die("truncated trace header");
  if (std::memcmp(header.magic, kTraceMagic, sizeof(kTraceMagic)) != 0 || header.version != kTraceVersion) {
    die("unsupported G2 trace magic/version");
  }
  if (header.dimension == 0 || header.base_n == 0 || header.pool_n < header.base_n ||
      header.reservoir_n != header.pool_n - header.base_n || header.query_n == 0 ||
      header.k == 0 || header.k > header.base_n || header.radius < 0.0F) die("invalid trace header");
  std::vector<TraceEvent> events(header.event_count);
  input.read(reinterpret_cast<char*>(events.data()),
             static_cast<std::streamsize>(events.size() * sizeof(TraceEvent)));
  if (input.gcount() != static_cast<std::streamsize>(events.size() * sizeof(TraceEvent))) die("truncated trace events");
  char extra = 0;
  if (input.read(&extra, 1)) die("extra bytes in trace");
  for (std::size_t index = 0; index < events.size(); ++index) {
    if (events[index].op_index != index || events[index].op < kInsert || events[index].op > kRebuild) {
      die("invalid trace event at op " + std::to_string(index));
    }
  }
  return {header, std::move(events)};
}

std::string trim_ascii(std::string value) {
  const std::size_t first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const std::size_t last = value.find_last_not_of(" \t\r\n");
  return value.substr(first, last - first + 1U);
}

int parse_contract_int(const std::unordered_map<std::string, std::string>& values, const std::string& key) {
  const auto found = values.find(key);
  if (found == values.end()) die("missing G2 contract field " + key);
  std::size_t parsed = 0;
  try {
    const int value = std::stoi(found->second, &parsed);
    if (parsed != found->second.size()) die("non-integer G2 contract field " + key);
    return value;
  } catch (const std::exception&) {
    die("invalid integer G2 contract field " + key);
  }
}

struct G2RebuildWitnessContract {
  int leaf_capacity = -1;
  int sidecar_leaf_id = -1;
  int a = -1;
  int b = -1;
  int c = -1;
  int query_before = -1;
  int query_middle = -1;
  int query_after = -1;
  int base_delete = -1;
  std::string v3_g1b_result_sha256;
  std::string v2_selection_sha256;
};

G2RebuildWitnessContract read_g2_rebuild_witness_contract(const std::string& path) {
  std::ifstream input(path);
  if (!input) die("cannot open G2 rebuild contract " + path);
  std::unordered_map<std::string, std::string> values;
  std::string line;
  while (std::getline(input, line)) {
    line = trim_ascii(line);
    if (line.empty() || line[0] == '#') continue;
    const std::size_t separator = line.find('=');
    if (separator == std::string::npos || line.find('=', separator + 1U) != std::string::npos) {
      die("malformed G2 rebuild contract line");
    }
    const std::string key = trim_ascii(line.substr(0, separator));
    const std::string value = trim_ascii(line.substr(separator + 1U));
    if (key.empty() || value.empty() || values.find(key) != values.end()) die("duplicate/empty G2 contract field");
    values.emplace(key, value);
  }
  static const std::vector<std::string> required = {
      "schema", "leaf_capacity", "sidecar_leaf_id", "stable_id_a", "stable_id_b", "stable_id_c",
      "query_id_before", "query_id_middle", "query_id_after", "base_delete_id",
      "v3_g1b_result_sha256", "v2_selection_sha256"};
  if (values.size() != required.size()) die("unexpected G2 contract field count");
  for (const std::string& key : required) if (values.find(key) == values.end()) die("missing G2 contract field " + key);
  if (values.at("schema") != "safe-c1-g2-rebuild-witness-contract-v4") die("wrong G2 rebuild contract schema");
  G2RebuildWitnessContract contract;
  contract.leaf_capacity = parse_contract_int(values, "leaf_capacity");
  contract.sidecar_leaf_id = parse_contract_int(values, "sidecar_leaf_id");
  contract.a = parse_contract_int(values, "stable_id_a");
  contract.b = parse_contract_int(values, "stable_id_b");
  contract.c = parse_contract_int(values, "stable_id_c");
  contract.query_before = parse_contract_int(values, "query_id_before");
  contract.query_middle = parse_contract_int(values, "query_id_middle");
  contract.query_after = parse_contract_int(values, "query_id_after");
  contract.base_delete = parse_contract_int(values, "base_delete_id");
  contract.v3_g1b_result_sha256 = values.at("v3_g1b_result_sha256");
  contract.v2_selection_sha256 = values.at("v2_selection_sha256");
  if (contract.leaf_capacity != 1 || contract.sidecar_leaf_id < 0 || contract.a < 0 || contract.b < 0 ||
      contract.c < 0 || contract.a == contract.b || contract.a == contract.c || contract.b == contract.c ||
      contract.query_before < 0 || contract.query_middle < 0 || contract.query_after < 0 ||
      contract.base_delete < 0 || contract.v3_g1b_result_sha256.size() != 64U ||
      contract.v2_selection_sha256.size() != 64U) die("invalid G2 rebuild contract values");
  return contract;
}

void validate_g2_rebuild_trace(const TraceHeader& header, const std::vector<TraceEvent>& events,
                               const G2RebuildWitnessContract& contract, int requested_leaf_capacity) {
  if (requested_leaf_capacity != 1 || requested_leaf_capacity != contract.leaf_capacity) {
    die("G2 witness requires leaf capacity exactly one");
  }
  if (contract.a < static_cast<int>(header.base_n) || contract.b < static_cast<int>(header.base_n) ||
      contract.c < static_cast<int>(header.base_n) || contract.a >= static_cast<int>(header.pool_n) ||
      contract.b >= static_cast<int>(header.pool_n) || contract.c >= static_cast<int>(header.pool_n) ||
      contract.base_delete >= static_cast<int>(header.base_n) ||
      contract.query_before >= static_cast<int>(header.query_n) ||
      contract.query_middle >= static_cast<int>(header.query_n) || contract.query_after >= static_cast<int>(header.query_n)) {
    die("G2 contract IDs outside expected initial-base/reservoir/query domains");
  }
  const std::vector<std::pair<int, int>> expected = {
      {kInsert, contract.a}, {kInsert, contract.b}, {kKnn, contract.query_before},
      {kDelete, contract.a}, {kInsert, contract.c}, {kKnn, contract.query_middle},
      {kDelete, contract.base_delete}, {kRebuild, 0}, {kKnn, contract.query_after}};
  if (events.size() != expected.size() || header.event_count != expected.size()) die("G2 trace must have exactly nine events");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    if (events[index].op_index != index || events[index].op != expected[index].first ||
        events[index].argument != expected[index].second) {
      die("trace does not match strict direct-delta-reuse-base-delete-immediate-rebuild sequence at op " +
          std::to_string(index));
    }
  }
}

struct MetricEncodingPlan {
  std::uint64_t bbox_squared_upper_bound = 0;
  long double bbox_l2_upper_bound = 0.0L;
  int infi_dis = 0;
  int dis_code = 100;
  bool raised_above_legacy_default = false;
};

MetricEncodingPlan make_metric_encoding_plan(const std::vector<std::int16_t>& pool,
                                              const std::vector<std::int16_t>& queries, int dimension) {
  constexpr int kLegacyInfiDis = 10000;
  constexpr int kDisCode = 100;
  constexpr long double kFloatSafetyMargin = 1024.0L;
  if (dimension <= 0 || pool.empty() || queries.empty() ||
      pool.size() % static_cast<std::size_t>(dimension) != 0 ||
      queries.size() % static_cast<std::size_t>(dimension) != 0) die("malformed metric inputs");
  std::vector<int> lower(dimension, std::numeric_limits<int>::max());
  std::vector<int> upper(dimension, std::numeric_limits<int>::min());
  auto incorporate = [&](const std::vector<std::int16_t>& values) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      const int axis = static_cast<int>(index % static_cast<std::size_t>(dimension));
      lower[axis] = std::min(lower[axis], static_cast<int>(values[index]));
      upper[axis] = std::max(upper[axis], static_cast<int>(values[index]));
    }
  };
  incorporate(pool); incorporate(queries);
  std::uint64_t squared = 0;
  for (int axis = 0; axis < dimension; ++axis) {
    const std::int64_t span = static_cast<std::int64_t>(upper[axis]) - lower[axis];
    const std::uint64_t term = static_cast<std::uint64_t>(span * span);
    if (squared > std::numeric_limits<std::uint64_t>::max() - term) die("metric bound overflow");
    squared += term;
  }
  const long double l2 = std::sqrt(static_cast<long double>(squared));
  const long double required = std::ceil(l2) + kFloatSafetyMargin;
  if (!std::isfinite(required) || required > static_cast<long double>(1 << 24)) die("metric sentinel invalid");
  MetricEncodingPlan plan;
  plan.bbox_squared_upper_bound = squared;
  plan.bbox_l2_upper_bound = l2;
  plan.infi_dis = std::max(kLegacyInfiDis, static_cast<int>(required));
  plan.dis_code = kDisCode;
  plan.raised_above_legacy_default = plan.infi_dis > kLegacyInfiDis;
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

const char* placement_label(safe_c1::PlacementKind placement) {
  switch (placement) {
    case safe_c1::PlacementKind::kBase: return "base";
    case safe_c1::PlacementKind::kDirect: return "direct";
    case safe_c1::PlacementKind::kDelta: return "delta";
    case safe_c1::PlacementKind::kDeleted: return "deleted";
    default: return "unknown";
  }
}

std::uint64_t fnv_ids(const std::vector<int>& ids) {
  return safe_c1::FrozenTreeSnapshot::fnv1a(
      ids.empty() ? nullptr : ids.data(), ids.size() * sizeof(int));
}

std::uint64_t fnv_active(const std::vector<std::uint8_t>& active) {
  return safe_c1::FrozenTreeSnapshot::fnv1a(
      active.empty() ? nullptr : active.data(), active.size() * sizeof(std::uint8_t));
}

std::string json_float(double value) {
  std::ostringstream out;
  out << std::setprecision(12) << value;
  return out.str();
}

void emit_int_array(std::ostream& output, const std::vector<int>& values) {
  output << '[';
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index) output << ',';
    output << values[index];
  }
  output << ']';
}

// A rebuild epoch has a new frozen geometry.  This wrapper proves the precise
// stable-ID leaf population, and records the exact pivot IDs actually used by
// the constructor.  Pivot IDs are required to be a subset of epoch live IDs;
// they are not (and must not be claimed to be) a second copy of every live ID.
struct EpochFrozen {
  safe_c1::FrozenTreeSnapshot snapshot;
  std::vector<int> expected_live_ids;
  std::vector<int> exact_leaf_ids;
  std::vector<int> exact_pivot_ids;
  std::uint64_t expected_live_hash = 0;
  std::uint64_t exact_leaf_hash = 0;
  std::uint64_t exact_pivot_hash = 0;

  static void require_sorted_unique(const std::vector<int>& ids, const char* label) {
    if (ids.empty() || !std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
      die(std::string("invalid sorted/unique ") + label);
    }
  }

  void capture(const safe_c1::BaseTreeRuntime& runtime,
               const std::vector<int>& expected) {
    require_sorted_unique(expected, "epoch expected stable IDs");
    if (static_cast<int>(expected.size()) != runtime.base_count) {
      die("epoch expected stable-ID cardinality disagrees with logical base count");
    }
    snapshot.capture(runtime);
    expected_live_ids = expected;
    expected_live_hash = fnv_ids(expected_live_ids);
    exact_leaf_ids.clear();
    exact_pivot_ids.clear();
    for (int node_id = 0; node_id < static_cast<int>(snapshot.nodes.size()); ++node_id) {
      if (snapshot.empty[static_cast<std::size_t>(node_id)] != 0) continue;
      const TN& node = snapshot.nodes[static_cast<std::size_t>(node_id)];
      if (node.pid >= 0) exact_pivot_ids.push_back(node.pid);
      if (node.is_leaf != 1) continue;
      if (node.lid < 0 || node.size < 0) die("epoch leaf has invalid logical range");
      std::vector<int> local(static_cast<std::size_t>(node.size));
      if (!local.empty()) {
        G2_CUDA(cudaMemcpy(local.data(), runtime.id_list + node.lid,
                           local.size() * sizeof(int), cudaMemcpyDeviceToHost));
        exact_leaf_ids.insert(exact_leaf_ids.end(), local.begin(), local.end());
      }
    }
    std::sort(exact_leaf_ids.begin(), exact_leaf_ids.end());
    std::sort(exact_pivot_ids.begin(), exact_pivot_ids.end());
    exact_pivot_ids.erase(std::unique(exact_pivot_ids.begin(), exact_pivot_ids.end()),
                          exact_pivot_ids.end());
    if (exact_leaf_ids != expected_live_ids) {
      die("epoch leaf stable-ID set is not exactly the requested active set");
    }
    for (int pivot : exact_pivot_ids) {
      if (!std::binary_search(expected_live_ids.begin(), expected_live_ids.end(), pivot)) {
        die("epoch pivot stable ID is absent from active seed set");
      }
    }
    exact_leaf_hash = fnv_ids(exact_leaf_ids);
    exact_pivot_hash = fnv_ids(exact_pivot_ids);
  }

  void assert_unchanged(const safe_c1::BaseTreeRuntime& runtime) const {
    EpochFrozen current;
    current.capture(runtime, expected_live_ids);
    if (current.snapshot.hash != snapshot.hash ||
        current.expected_live_hash != expected_live_hash ||
        current.exact_leaf_hash != exact_leaf_hash ||
        current.exact_pivot_hash != exact_pivot_hash ||
        current.exact_pivot_ids != exact_pivot_ids) {
      die("frozen epoch tree changed before an explicit rebuild");
    }
  }
};

struct InsertReceipt {
  safe_c1::PlacementKind placement = safe_c1::PlacementKind::kUnknown;
  int certified_leaf_id = -1;
  InsertFallbackReason fallback_reason = InsertFallbackReason::kNone;
};

// Host-only metadata.  It intentionally has no pointer to legacy TN/id_list;
// direct objects remain a sidecar and delta objects remain an exact global tier.
// A base deletion creates a hard barrier: the only next legal trace operation
// is REBUILD, so no query can accidentally combine a stale base tree with a
// base-deleted active set.
class G2IntegrationState {
 public:
  G2IntegrationState(int pool_n, const std::vector<int>& initial_base,
                     int leaf_capacity)
      : active_(static_cast<std::size_t>(pool_n), 0),
        placement_(static_cast<std::size_t>(pool_n), safe_c1::PlacementKind::kDeleted),
        sidecar_leaf_for_(static_cast<std::size_t>(pool_n), -1),
        certified_leaf_for_(static_cast<std::size_t>(pool_n), -1),
        fallback_reason_for_(static_cast<std::size_t>(pool_n), InsertFallbackReason::kNone),
        leaf_capacity_(leaf_capacity) {
    if (pool_n <= 0 || leaf_capacity <= 0) die("invalid G2 state dimensions");
    initialize_base(initial_base);
  }

  void initialize_base(const std::vector<int>& ids) {
    if (!std::is_sorted(ids.begin(), ids.end()) ||
        std::adjacent_find(ids.begin(), ids.end()) != ids.end()) {
      die("G2 base seed is not sorted/unique");
    }
    for (int id : ids) {
      check(id);
      if (active_[static_cast<std::size_t>(id)] != 0) die("duplicate G2 base stable ID");
      active_[static_cast<std::size_t>(id)] = 1;
      placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kBase;
    }
  }

  InsertReceipt insert(int id, const EpochFrozen& frozen,
                       const safe_c1::HostVectorPool& pool) {
    require_not_rebuild_pending("insert");
    check(id);
    if (active_[static_cast<std::size_t>(id)] != 0) die("insert targets active stable ID");
    active_[static_cast<std::size_t>(id)] = 1;
    const int certified_leaf = frozen.snapshot.certify_leaf(pool, id);
    certified_leaf_for_[static_cast<std::size_t>(id)] = certified_leaf;
    sidecar_leaf_for_[static_cast<std::size_t>(id)] = -1;
    if (certified_leaf >= 0 &&
        static_cast<int>(sidecars_[certified_leaf].size()) < leaf_capacity_) {
      sidecars_[certified_leaf].push_back(id);
      placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDirect;
      sidecar_leaf_for_[static_cast<std::size_t>(id)] = certified_leaf;
      fallback_reason_for_[static_cast<std::size_t>(id)] = InsertFallbackReason::kNone;
      ++direct_inserts_;
      return {safe_c1::PlacementKind::kDirect, certified_leaf, InsertFallbackReason::kNone};
    }
    placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDelta;
    delta_.push_back(id);
    const InsertFallbackReason reason = certified_leaf < 0
        ? InsertFallbackReason::kCertificateReject : InsertFallbackReason::kCapacityFull;
    fallback_reason_for_[static_cast<std::size_t>(id)] = reason;
    if (reason == InsertFallbackReason::kCertificateReject) ++certificate_rejects_;
    else ++capacity_rejects_;
    ++delta_inserts_;
    return {safe_c1::PlacementKind::kDelta, certified_leaf, reason};
  }

  void erase(int id) {
    check(id);
    if (active_[static_cast<std::size_t>(id)] == 0) die("delete targets inactive stable ID");
    if (rebuild_required_) die("only REBUILD may follow a base deletion");
    const safe_c1::PlacementKind prior = placement_[static_cast<std::size_t>(id)];
    if (prior == safe_c1::PlacementKind::kDirect) {
      const int leaf = sidecar_leaf_for_[static_cast<std::size_t>(id)];
      auto found = sidecars_.find(leaf);
      if (found == sidecars_.end()) die("missing direct sidecar placement");
      auto& values = found->second;
      values.erase(std::remove(values.begin(), values.end(), id), values.end());
    } else if (prior == safe_c1::PlacementKind::kDelta) {
      delta_.erase(std::remove(delta_.begin(), delta_.end(), id), delta_.end());
    } else if (prior == safe_c1::PlacementKind::kBase) {
      rebuild_required_ = true;
      base_deletes_++;
    } else {
      die("active ID has invalid placement");
    }
    active_[static_cast<std::size_t>(id)] = 0;
    placement_[static_cast<std::size_t>(id)] = safe_c1::PlacementKind::kDeleted;
    sidecar_leaf_for_[static_cast<std::size_t>(id)] = -1;
  }

  void require_not_rebuild_pending(const char* operation) const {
    if (rebuild_required_) die(std::string("base deletion requires immediate REBUILD before ") + operation);
  }

  void require_rebuild_pending() const {
    if (!rebuild_required_) die("REBUILD appears without a preceding base deletion");
  }

  void after_rebuild(const std::vector<int>& newly_seeded_live_ids) {
    require_rebuild_pending();
    const std::vector<int> before = live_ids();
    if (before != newly_seeded_live_ids) {
      die("rebuild seed stable-ID set does not preserve active set");
    }
    sidecars_.clear();
    delta_.clear();
    for (int id = 0; id < static_cast<int>(active_.size()); ++id) {
      const std::size_t position = static_cast<std::size_t>(id);
      sidecar_leaf_for_[position] = -1;
      certified_leaf_for_[position] = -1;
      fallback_reason_for_[position] = InsertFallbackReason::kNone;
      placement_[position] = active_[position] != 0
          ? safe_c1::PlacementKind::kBase : safe_c1::PlacementKind::kDeleted;
    }
    rebuild_required_ = false;
    ++rebuilds_;
  }

  const std::vector<std::uint8_t>& active() const { return active_; }
  std::vector<int> live_ids() const {
    std::vector<int> ids;
    for (int id = 0; id < static_cast<int>(active_.size()); ++id)
      if (active_[static_cast<std::size_t>(id)] != 0) ids.push_back(id);
    return ids;
  }
  std::vector<int> direct_ids() const {
    std::vector<int> ids;
    for (int id = 0; id < static_cast<int>(placement_.size()); ++id)
      if (placement_[static_cast<std::size_t>(id)] == safe_c1::PlacementKind::kDirect) ids.push_back(id);
    return ids;
  }
  const std::vector<int>& delta_ids() const { return delta_; }
  std::vector<int> sidecar_candidates_for(const std::vector<int>& visited_leaf_ids) const {
    std::vector<int> ids;
    for (int leaf : visited_leaf_ids) {
      const auto found = sidecars_.find(leaf);
      if (found == sidecars_.end()) continue;
      for (int id : found->second) {
        if (active_[static_cast<std::size_t>(id)] != 0 &&
            placement_[static_cast<std::size_t>(id)] == safe_c1::PlacementKind::kDirect) ids.push_back(id);
      }
    }
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    return ids;
  }
  safe_c1::PlacementKind placement_of(int id) const { check(id); return placement_[static_cast<std::size_t>(id)]; }
  int leaf_of(int id) const { check(id); return sidecar_leaf_for_[static_cast<std::size_t>(id)]; }
  int certified_leaf_of(int id) const { check(id); return certified_leaf_for_[static_cast<std::size_t>(id)]; }
  InsertFallbackReason fallback_of(int id) const { check(id); return fallback_reason_for_[static_cast<std::size_t>(id)]; }
  bool rebuild_required() const { return rebuild_required_; }
  int active_count() const { return static_cast<int>(live_ids().size()); }
  int direct_inserts() const { return direct_inserts_; }
  int delta_inserts() const { return delta_inserts_; }
  int capacity_rejects() const { return capacity_rejects_; }
  int certificate_rejects() const { return certificate_rejects_; }
  int base_deletes() const { return base_deletes_; }
  int rebuilds() const { return rebuilds_; }
  std::uint64_t active_hash() const { return fnv_active(active_); }

 private:
  void check(int id) const {
    if (id < 0 || id >= static_cast<int>(active_.size())) die("stable ID outside full immutable pool");
  }
  std::vector<std::uint8_t> active_;
  std::vector<safe_c1::PlacementKind> placement_;
  std::vector<int> sidecar_leaf_for_;
  std::vector<int> certified_leaf_for_;
  std::vector<InsertFallbackReason> fallback_reason_for_;
  int leaf_capacity_ = 0;
  std::unordered_map<int, std::vector<int>> sidecars_;
  std::vector<int> delta_;
  bool rebuild_required_ = false;
  int direct_inserts_ = 0;
  int delta_inserts_ = 0;
  int certificate_rejects_ = 0;
  int capacity_rejects_ = 0;
  int base_deletes_ = 0;
  int rebuilds_ = 0;
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
    const double difference = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
    sum += difference * difference;
  }
  distances[position] = sqrt(sum);
}

std::uint64_t quantized_squared_l2(const std::vector<std::int16_t>& pool,
                                    const std::vector<std::int16_t>& queries,
                                    int dimension, int stable_id, int query_id) {
  if (stable_id < 0 || query_id < 0) die("negative ID in exact tie check");
  const std::size_t point = static_cast<std::size_t>(stable_id) * dimension;
  const std::size_t query = static_cast<std::size_t>(query_id) * dimension;
  if (point + dimension > pool.size() || query + dimension > queries.size()) die("tie check ID out of range");
  std::uint64_t sum = 0;
  for (int d = 0; d < dimension; ++d) {
    const std::int64_t delta = static_cast<std::int64_t>(pool[point + d]) -
                               static_cast<std::int64_t>(queries[query + d]);
    const std::uint64_t term = static_cast<std::uint64_t>(delta * delta);
    if (sum > std::numeric_limits<std::uint64_t>::max() - term) die("exact tie key overflow");
    sum += term;
  }
  return sum;
}

float modeled_gts_l2(const std::vector<std::int16_t>& pool,
                     const std::vector<std::int16_t>& queries,
                     int dimension, int stable_id, int query_id) {
  const std::size_t point = static_cast<std::size_t>(stable_id) * dimension;
  const std::size_t query = static_cast<std::size_t>(query_id) * dimension;
  double sum = 0.0;
  for (int d = 0; d < dimension; ++d) {
    const float delta = static_cast<float>(pool[point + d]) - static_cast<float>(queries[query + d]);
    sum += static_cast<double>(delta * delta);
  }
  return std::sqrt(static_cast<float>(sum));
}

struct TieAudit { int static_queries = 0; int dynamic_boundaries = 0; };

void require_static_epoch_tie_free(const std::vector<int>& epoch_ids,
                                   const std::vector<std::int16_t>& pool,
                                   const std::vector<std::int16_t>& queries,
                                   int dimension, int k, TieAudit* audit) {
  if (static_cast<int>(epoch_ids.size()) < k) die("epoch below requested static k");
  const int qnum = static_cast<int>(queries.size() / static_cast<std::size_t>(dimension));
  for (int qid = 0; qid < qnum; ++qid) {
    std::vector<std::pair<std::uint64_t, int>> exact;
    std::vector<std::pair<float, int>> modeled;
    exact.reserve(epoch_ids.size()); modeled.reserve(epoch_ids.size());
    for (int id : epoch_ids) {
      exact.emplace_back(quantized_squared_l2(pool, queries, dimension, id, qid), id);
      modeled.emplace_back(modeled_gts_l2(pool, queries, dimension, id, qid), id);
    }
    std::sort(exact.begin(), exact.end()); std::sort(modeled.begin(), modeled.end());
    const int prefix = std::min(static_cast<int>(epoch_ids.size()), k + 1);
    for (int rank = 1; rank < prefix; ++rank) {
      if (exact[static_cast<std::size_t>(rank - 1)].first == exact[static_cast<std::size_t>(rank)].first ||
          modeled[static_cast<std::size_t>(rank - 1)].first == modeled[static_cast<std::size_t>(rank)].first) {
        die("ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN: static epoch prefix tie");
      }
    }
    for (int rank = 0; rank < k; ++rank) {
      if (exact[static_cast<std::size_t>(rank)].second != modeled[static_cast<std::size_t>(rank)].second) {
        die("ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN: exact/modelled static epoch order mismatch");
      }
    }
    if (audit) ++audit->static_queries;
  }
}

void require_dynamic_boundary_tie_free(const std::vector<std::uint8_t>& active,
                                       const std::vector<std::int16_t>& pool,
                                       const std::vector<std::int16_t>& queries,
                                       int dimension, int k, int qid, TieAudit* audit) {
  std::vector<std::pair<std::uint64_t, int>> values;
  for (int id = 0; id < static_cast<int>(active.size()); ++id) if (active[static_cast<std::size_t>(id)] != 0)
    values.emplace_back(quantized_squared_l2(pool, queries, dimension, id, qid), id);
  std::sort(values.begin(), values.end());
  if (static_cast<int>(values.size()) < k) die("active set below k");
  if (static_cast<int>(values.size()) > k &&
      values[static_cast<std::size_t>(k - 1)].first == values[static_cast<std::size_t>(k)].first) {
    die("ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN: dynamic k boundary tie");
  }
  if (audit) ++audit->dynamic_boundaries;
}

bool close_distance(float left, float right) {
  const float magnitude = std::max({1.0F, std::fabs(left), std::fabs(right)});
  const float ulp = std::nextafter(magnitude, std::numeric_limits<float>::infinity()) - magnitude;
  return std::fabs(left - right) <= std::max(kProbeDistanceAbsoluteFloor,
                                              static_cast<float>(kProbeDistanceMaxFloatUlps) * ulp);
}

void validate_real_static_probe(safe_c1::BaseTreeRuntime& runtime, float* queries_d,
                                const std::vector<int>& epoch_ids,
                                const std::vector<std::int16_t>& pool,
                                const std::vector<std::int16_t>& queries,
                                int dimension, int k) {
  const int qnum = static_cast<int>(queries.size() / static_cast<std::size_t>(dimension));
  const safe_c1::TopKPairs observed = safe_c1::probe_static_base_topk(runtime, queries_d, qnum, k);
  if (observed.ids.size() != static_cast<std::size_t>(qnum) * k ||
      observed.distances.size() != static_cast<std::size_t>(qnum) * k) {
    die("real GTS static probe result shape mismatch");
  }
  for (int qid = 0; qid < qnum; ++qid) {
    std::vector<std::pair<std::uint64_t, int>> exact;
    exact.reserve(epoch_ids.size());
    for (int id : epoch_ids) exact.emplace_back(quantized_squared_l2(pool, queries, dimension, id, qid), id);
    std::sort(exact.begin(), exact.end());
    std::vector<std::pair<std::uint64_t, int>> observed_key;
    for (int rank = 0; rank < k; ++rank) {
      const std::size_t pos = static_cast<std::size_t>(qid) * k + rank;
      const int id = observed.ids[pos];
      if (!std::binary_search(epoch_ids.begin(), epoch_ids.end(), id)) die("GTS static probe emitted ID outside epoch");
      const float reference = modeled_gts_l2(pool, queries, dimension, id, qid);
      if (!close_distance(observed.distances[pos], reference)) die("GTS static probe distance mismatch");
      observed_key.emplace_back(quantized_squared_l2(pool, queries, dimension, id, qid), id);
    }
    std::sort(observed_key.begin(), observed_key.end());
    std::vector<std::pair<std::uint64_t, int>> expected(exact.begin(), exact.begin() + k);
    if (observed_key != expected) die("GTS static probe exact top-k ID-set mismatch");
  }
}

void release_tree_metadata_only(safe_c1::BaseTreeRuntime& runtime) {
  if (runtime.id_list) { G2_CUDA(cudaFree(runtime.id_list)); runtime.id_list = nullptr; }
  if (runtime.node_list) { G2_CUDA(cudaFree(runtime.node_list)); runtime.node_list = nullptr; }
  if (runtime.max_node_num) { G2_CUDA(cudaFree(runtime.max_node_num)); runtime.max_node_num = nullptr; }
  if (runtime.empty_list) { G2_CUDA(cudaFree(runtime.empty_list)); runtime.empty_list = nullptr; }
  if (max_dis_d) { G2_CUDA(cudaFree(max_dis_d)); max_dis_d = nullptr; }
  if (dis_list) { G2_CUDA(cudaFree(dis_list)); dis_list = nullptr; }
  if (split_list) { G2_CUDA(cudaFree(split_list)); split_list = nullptr; }
  if (split_num) { G2_CUDA(cudaFree(split_num)); split_num = nullptr; }
  if (pid_list) { G2_CUDA(cudaFree(pid_list)); pid_list = nullptr; }
  runtime.tree_height = 0;
  runtime.base_count = 0;
}

void release_tree_full(safe_c1::BaseTreeRuntime& runtime) {
  release_tree_metadata_only(runtime);
  if (runtime.data_d) { G2_CUDA(cudaFree(runtime.data_d)); runtime.data_d = nullptr; }
  if (runtime.data_info) { G2_CUDA(cudaFree(runtime.data_info)); runtime.data_info = nullptr; }
}

void build_seeded_epoch(safe_c1::BaseTreeRuntime& runtime, const std::vector<int>& live_ids) {
  if (runtime.data_d == nullptr || runtime.data_info == nullptr) die("missing immutable full pool before rebuild");
  if (live_ids.empty() || !std::is_sorted(live_ids.begin(), live_ids.end()) ||
      std::adjacent_find(live_ids.begin(), live_ids.end()) != live_ids.end()) die("invalid rebuild live stable-ID seed");
  runtime.data_info[1] = static_cast<int>(live_ids.size());
  indexConstruSeededStableIds(runtime.data_d, nullptr, nullptr, runtime.data_info, live_ids,
                              runtime.id_list, runtime.node_list, runtime.max_node_num,
                              runtime.tree_height, runtime.empty_list);
  G2_CUDA(cudaDeviceSynchronize());
  G2_CUDA(cudaGetLastError());
  runtime.base_count = static_cast<int>(live_ids.size());
  if (!runtime.ready()) die("seeded GTS epoch constructor did not produce runtime");
}

std::vector<std::pair<double, int>> exact_active_oracle(const safe_c1::HostVectorPool& pool,
                                                         const float* query,
                                                         const std::vector<std::uint8_t>& active,
                                                         int k) {
  std::vector<std::pair<double, int>> rows;
  for (int id = 0; id < static_cast<int>(active.size()); ++id) {
    if (active[static_cast<std::size_t>(id)] == 0) continue;
    const float* vector = pool.at(id);
    double total = 0.0;
    for (int d = 0; d < pool.dimension; ++d) {
      const double delta = static_cast<double>(vector[d]) - static_cast<double>(query[d]);
      total += delta * delta;
    }
    rows.emplace_back(std::sqrt(total), id);
  }
  std::sort(rows.begin(), rows.end(), [](const auto& left, const auto& right) {
    return left.first != right.first ? left.first < right.first : left.second < right.second;
  });
  if (static_cast<int>(rows.size()) < k) die("exact active oracle has fewer than k live IDs");
  rows.resize(static_cast<std::size_t>(k));
  return rows;
}

void require_same_exact_rows(const std::vector<std::pair<double, int>>& candidate,
                             const std::vector<std::pair<double, int>>& oracle) {
  if (candidate.size() != oracle.size()) die("dynamic candidate result cardinality mismatch");
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    if (candidate[i].second != oracle[i].second ||
        std::fabs(candidate[i].first - oracle[i].first) > 1.0e-8 * std::max(1.0, std::fabs(oracle[i].first))) {
      die("dynamic selective GTS/sidecar/delta merge disagrees with independent full active-set oracle");
    }
  }
}

}  // close outer anonymous namespace from part1

int main(int argc, char** argv) {
  safe_c1::BaseTreeRuntime runtime;
  float* query_d = nullptr;
  int* candidate_ids_d = nullptr;
  double* candidate_distances_d = nullptr;
  try {
    const Args args = parse_args(argc, argv);
    if (args.rebuild_contract != args.bundle + "/g2_rebuild_contract.txt") {
      die("G2 rebuild contract must be bundle-local");
    }
    const auto trace = read_trace(args.bundle + "/trace.g2trc");
    const TraceHeader& header = trace.first;
    const std::vector<TraceEvent>& events = trace.second;
    const G2RebuildWitnessContract contract =
        read_g2_rebuild_witness_contract(args.rebuild_contract);
    validate_g2_rebuild_trace(header, events, contract, args.leaf_capacity);

    // No CUDA API before this complete CPU semantic preflight.
    const std::vector<int> initial_ids = require_identity_layout(args.bundle, header);
    const std::size_t pool_values = static_cast<std::size_t>(header.pool_n) * header.dimension;
    const std::size_t query_values = static_cast<std::size_t>(header.query_n) * header.dimension;
    const std::vector<std::int16_t> pool_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/pool.i16", pool_values);
    const std::vector<std::int16_t> queries_i16 =
        read_exact_binary<std::int16_t>(args.bundle + "/queries.i16", query_values);
    const MetricEncodingPlan metric = make_metric_encoding_plan(
        pool_i16, queries_i16, static_cast<int>(header.dimension));
    TieAudit ties;
    require_static_epoch_tie_free(initial_ids, pool_i16, queries_i16,
                                  static_cast<int>(header.dimension), static_cast<int>(header.k), &ties);

    safe_c1::HostVectorPool pool;
    pool.dimension = static_cast<int>(header.dimension);
    pool.vectors.resize(pool_values);
    for (std::size_t i = 0; i < pool_values; ++i) pool.vectors[i] = static_cast<float>(pool_i16[i]);
    std::vector<float> query_host(query_values);
    for (std::size_t i = 0; i < query_values; ++i) query_host[i] = static_cast<float>(queries_i16[i]);
    // Hash the precise physical encoding copied to data_d, rather than the
    // separately maintained HostVectorPool view.  `GtsScalar` is asserted
    // above to be the archive float encoding in this build.
    std::vector<GtsScalar> immutable_pool_encoded(pool_values);
    for (std::size_t i = 0; i < pool_values; ++i) {
      immutable_pool_encoded[i] = static_cast<GtsScalar>(pool_i16[i]);
    }
    const std::uint64_t host_pool_hash = safe_c1::FrozenTreeSnapshot::fnv1a(
        immutable_pool_encoded.data(), immutable_pool_encoded.size() * sizeof(GtsScalar));

    DIS_CODE = metric.dis_code;
    INFI_DIS = metric.infi_dis;
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_info), 3 * sizeof(int)));
    runtime.data_info[0] = static_cast<int>(header.dimension);
    runtime.data_info[1] = static_cast<int>(header.base_n);
    runtime.data_info[2] = 2;
    // One immutable physical pool for both epochs; logical membership is seeded IDs.
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&runtime.data_d), pool_values * sizeof(GtsScalar)));
    for (std::size_t i = 0; i < pool_values; ++i) runtime.data_d[i] = immutable_pool_encoded[i];
    float alpha[RP_MAX_LEVELS]{};
    float beta[RP_MAX_LEVELS]{};
    float gamma[RP_MAX_LEVELS];
    std::fill(std::begin(gamma), std::end(gamma), 1.0F);
    upload_rp_constants(alpha, beta, gamma, RP_MAX_LEVELS, nullptr, nullptr, nullptr, 0, 0);
    G2_CUDA(cudaMallocManaged(reinterpret_cast<void**>(&query_d), query_values * sizeof(float)));
    std::copy(query_host.begin(), query_host.end(), query_d);
    G2_CUDA(cudaMalloc(reinterpret_cast<void**>(&candidate_ids_d), header.pool_n * sizeof(int)));
    G2_CUDA(cudaMalloc(reinterpret_cast<void**>(&candidate_distances_d), header.pool_n * sizeof(double)));

    auto device_pool_hash = [&]() {
      std::vector<GtsScalar> copy(pool_values);
      G2_CUDA(cudaMemcpy(copy.data(), runtime.data_d,
                          copy.size() * sizeof(GtsScalar), cudaMemcpyDeviceToHost));
      return safe_c1::FrozenTreeSnapshot::fnv1a(copy.data(), copy.size() * sizeof(GtsScalar));
    };
    if (device_pool_hash() != host_pool_hash) die("full immutable pool mismatch before E0");

    build_seeded_epoch(runtime, initial_ids);
    EpochFrozen epoch0;
    epoch0.capture(runtime, initial_ids);
    validate_real_static_probe(runtime, query_d, initial_ids, pool_i16, queries_i16,
                               static_cast<int>(header.dimension), static_cast<int>(header.k));
    epoch0.assert_unchanged(runtime);

    G2IntegrationState state(static_cast<int>(header.pool_n), initial_ids, args.leaf_capacity);
    std::ofstream out(args.output);
    if (!out) die("cannot open G2 JSONL output");
    out << "{\"record\":\"meta\",\"schema\":\"safe-c1-g2-rebuild-witness-v4\","
        << "\"scope\":\"strict stable-ID immediate rebuild witness only; no timing/performance/general rebuild/range/C2/all-input tie claim\","
        << "\"legacy_incremental_updater_used\":false,\"stable_id_layout\":\"identity_full_immutable_pool_seeded_stable_ids_v4\","
        << "\"host_full_pool_hash\":" << host_pool_hash << "}\n";
    out << "{\"record\":\"epoch\",\"epoch\":0,\"phase\":\"E0\",\"tree_hash\":" << epoch0.snapshot.hash
        << ",\"leaf_stable_id_hash\":" << epoch0.exact_leaf_hash
        << ",\"pivot_stable_id_hash\":" << epoch0.exact_pivot_hash
        << ",\"live_stable_id_hash\":" << epoch0.expected_live_hash
        << ",\"live_count\":" << epoch0.expected_live_ids.size()
        << ",\"leaf_set_exact\":true,\"pivot_set_subset_of_live\":true,\"pivot_stable_ids\":";
    emit_int_array(out, epoch0.exact_pivot_ids); out << "}\n";

    auto emit_update = [&](const TraceEvent& e, safe_c1::PlacementKind p, int leaf, int certified,
                           InsertFallbackReason fallback) {
      out << "{\"record\":\"update\",\"op_index\":" << e.op_index
          << ",\"op\":\"" << (e.op == kInsert ? "insert" : "delete")
          << "\",\"stable_id\":" << e.argument << ",\"placement\":\"" << placement_label(p)
          << "\",\"sidecar_leaf_id\":" << leaf << ",\"certified_leaf_id\":" << certified
          << ",\"fallback_reason\":\"" << fallback_reason_label(fallback) << "\"}\n";
    };

    auto run_query = [&](const TraceEvent& e, const EpochFrozen& epoch,
                         const std::vector<int>& expected_sidecars,
                         const std::vector<int>& expected_delta, int expected_top1) {
      state.require_not_rebuild_pending("query");
      const int qid = e.argument;
      require_dynamic_boundary_tie_free(state.active(), pool_i16, queries_i16,
                                        static_cast<int>(header.dimension), static_cast<int>(header.k), qid, &ties);
      float* query = query_d + static_cast<std::size_t>(qid) * header.dimension;
      const safe_c1::TopKWithTraversalReceipt base = safe_c1::run_gts_base_topk_with_receipt(
          runtime, query, 1, static_cast<int>(header.k));
      epoch.assert_unchanged(runtime);
      std::vector<int> sidecars = state.sidecar_candidates_for(base.visited_leaf_ids);
      std::vector<int> extra = sidecars;
      extra.insert(extra.end(), state.delta_ids().begin(), state.delta_ids().end());
      std::sort(extra.begin(), extra.end());
      if (std::adjacent_find(extra.begin(), extra.end()) != extra.end()) die("overlapping sidecar/delta IDs");
      if (sidecars != expected_sidecars || state.delta_ids() != expected_delta) {
        die("unexpected selective tiers for strict G2 query");
      }
      for (int id : extra) {
        if (state.active()[static_cast<std::size_t>(id)] == 0 ||
            state.placement_of(id) == safe_c1::PlacementKind::kBase) die("invalid extra candidate");
      }
      std::vector<double> distances(extra.size());
      if (!extra.empty()) {
        G2_CUDA(cudaMemcpy(candidate_ids_d, extra.data(), extra.size() * sizeof(int), cudaMemcpyHostToDevice));
        exact_candidate_l2<<<(static_cast<int>(extra.size()) + 255) / 256, 256>>>(
            runtime.data_d, query, candidate_ids_d, candidate_distances_d,
            static_cast<int>(extra.size()), static_cast<int>(header.dimension));
        G2_CUDA(cudaGetLastError()); G2_CUDA(cudaDeviceSynchronize());
        G2_CUDA(cudaMemcpy(distances.data(), candidate_distances_d, distances.size() * sizeof(double),
                            cudaMemcpyDeviceToHost));
      }
      std::vector<std::pair<double, int>> merged;
      merged.reserve(static_cast<std::size_t>(header.k) + extra.size());
      for (int id : base.topk.ids) {
        if (id < 0 || id >= static_cast<int>(header.pool_n) ||
            state.active()[static_cast<std::size_t>(id)] == 0 ||
            state.placement_of(id) != safe_c1::PlacementKind::kBase) {
          die("GTS emitted non-base/deleted dynamic candidate");
        }
        const float* x = pool.at(id);
        double sum = 0.0;
        for (int d = 0; d < pool.dimension; ++d) {
          const double z = static_cast<double>(x[d]) - static_cast<double>(query[d]);
          sum += z * z;
        }
        merged.emplace_back(std::sqrt(sum), id);
      }
      for (std::size_t i = 0; i < extra.size(); ++i) merged.emplace_back(distances[i], extra[i]);
      std::sort(merged.begin(), merged.end(), [](const auto& a, const auto& b) {
        return a.first != b.first ? a.first < b.first : a.second < b.second;
      });
      if (merged.size() > header.k) merged.resize(header.k);
      const auto oracle = exact_active_oracle(pool, query, state.active(), static_cast<int>(header.k));
      require_same_exact_rows(merged, oracle);
      if (merged.empty() || merged.front().second != expected_top1) die("strict self-query top1 mismatch");
      out << "{\"record\":\"query\",\"op_index\":" << e.op_index << ",\"query_id\":" << qid
          << ",\"gts_visited_leaf_ids\":"; emit_int_array(out, base.visited_leaf_ids);
      out << ",\"sidecar_ids\":"; emit_int_array(out, sidecars);
      out << ",\"delta_ids\":"; emit_int_array(out, state.delta_ids());
      out << ",\"oracle_full_active_set_checked\":true,\"results\":[";
      for (std::size_t i = 0; i < merged.size(); ++i) {
        if (i) out << ',';
        out << '[' << merged[i].second << ',' << json_float(merged[i].first) << ']';
      }
      out << "]}\n";
    };

    EpochFrozen epoch1;
    bool have_e1 = false;
    std::uint64_t before_active_hash = 0, after_active_hash = 0, after_pool_hash = 0;
    for (const TraceEvent& e : events) {
      if (e.op == kInsert) {
        const InsertReceipt r = state.insert(e.argument, epoch0, pool);
        epoch0.assert_unchanged(runtime);
        if (e.op_index == 0) {
          if (r.placement != safe_c1::PlacementKind::kDirect ||
              r.certified_leaf_id != contract.sidecar_leaf_id ||
              r.fallback_reason != InsertFallbackReason::kNone ||
              state.leaf_of(e.argument) != contract.sidecar_leaf_id) {
            die("op0 requires direct a on selected leaf");
          }
        } else if (e.op_index == 1) {
          if (r.placement != safe_c1::PlacementKind::kDelta ||
              r.certified_leaf_id != contract.sidecar_leaf_id ||
              r.fallback_reason != InsertFallbackReason::kCapacityFull || state.leaf_of(e.argument) != -1) {
            die("op1 requires certified b capacity-delta");
          }
        } else if (e.op_index == 4) {
          if (r.placement != safe_c1::PlacementKind::kDirect ||
              r.certified_leaf_id != contract.sidecar_leaf_id ||
              r.fallback_reason != InsertFallbackReason::kNone ||
              state.leaf_of(e.argument) != contract.sidecar_leaf_id) {
            die("op4 requires direct c slot-reuse");
          }
        } else {
          die("unexpected insert event");
        }
        emit_update(e, r.placement, state.leaf_of(e.argument), r.certified_leaf_id, r.fallback_reason);
      } else if (e.op == kKnn) {
        if (e.op_index == 2) {
          run_query(e, epoch0, std::vector<int>{contract.a}, std::vector<int>{contract.b}, contract.a);
        } else if (e.op_index == 5) {
          run_query(e, epoch0, std::vector<int>{contract.c}, std::vector<int>{contract.b}, contract.c);
        } else if (e.op_index == 8) {
          if (!have_e1) die("post-rebuild query before E1");
          run_query(e, epoch1, std::vector<int>{}, std::vector<int>{}, contract.c);
        } else {
          die("unexpected kNN event");
        }
      } else if (e.op == kDelete) {
        const auto prior = state.placement_of(e.argument);
        const int leaf = state.leaf_of(e.argument);
        const int certificate = state.certified_leaf_of(e.argument);
        const auto fallback = state.fallback_of(e.argument);
        if (e.op_index == 3) {
          if (prior != safe_c1::PlacementKind::kDirect || leaf != contract.sidecar_leaf_id ||
              certificate != contract.sidecar_leaf_id || fallback != InsertFallbackReason::kNone) {
            die("op3 must delete direct a by recorded placement");
          }
          state.erase(e.argument);
          epoch0.assert_unchanged(runtime);
        } else if (e.op_index == 6) {
          if (prior != safe_c1::PlacementKind::kBase) die("op6 must delete base object");
          state.erase(e.argument);
          if (!state.rebuild_required()) die("base delete did not set rebuild barrier");
          epoch0.assert_unchanged(runtime);
        } else {
          die("unexpected delete event");
        }
        emit_update(e, safe_c1::PlacementKind::kDeleted, leaf, certificate, fallback);
      } else if (e.op == kRebuild) {
        if (e.op_index != 7 || e.argument != 0) die("unexpected rebuild event");
        state.require_rebuild_pending();
        epoch0.assert_unchanged(runtime);
        const std::vector<int> live = state.live_ids();
        const std::vector<int> direct_before = state.direct_ids();
        const std::vector<int> delta_before = state.delta_ids();
        if (direct_before != std::vector<int>{contract.c} || delta_before != std::vector<int>{contract.b}) {
          die("rebuild must receive c direct and b capacity-delta transient tiers");
        }
        before_active_hash = state.active_hash();
        const std::uint64_t pool_hash_before = device_pool_hash();
        if (pool_hash_before != host_pool_hash) die("immutable pool mutated before rebuild");

        // Exact required transition: free tree metadata only; retain data_d;
        // reseed all current active stable IDs before its first GTS pivot pass.
        release_tree_metadata_only(runtime);
        build_seeded_epoch(runtime, live);
        epoch1.capture(runtime, live);
        require_static_epoch_tie_free(live, pool_i16, queries_i16,
                                      static_cast<int>(header.dimension), static_cast<int>(header.k), &ties);
        validate_real_static_probe(runtime, query_d, live, pool_i16, queries_i16,
                                   static_cast<int>(header.dimension), static_cast<int>(header.k));
        epoch1.assert_unchanged(runtime);
        after_pool_hash = device_pool_hash();
        if (after_pool_hash != host_pool_hash) die("metadata-only rebuild mutated immutable full pool");
        state.after_rebuild(live);
        after_active_hash = state.active_hash();
        if (before_active_hash != after_active_hash || !state.direct_ids().empty() ||
            !state.delta_ids().empty() || state.rebuild_required()) {
          die("rebuild failed to preserve active set/clear transient tiers");
        }
        have_e1 = true;
        out << "{\"record\":\"rebuild\",\"op_index\":7,\"trigger\":\"base_delete_immediate\","
            << "\"metadata_only_release\":true,\"legacy_incremental_updater_used\":false,"
            << "\"seeded_base_stable_ids\":";
        emit_int_array(out, live);
        out << ",\"pre_rebuild_active_hash\":" << before_active_hash
            << ",\"post_rebuild_active_hash\":" << after_active_hash
            << ",\"immutable_pool_hash_before\":" << pool_hash_before
            << ",\"immutable_pool_hash_after\":" << after_pool_hash
            << ",\"transient_direct_before\":";
        emit_int_array(out, direct_before);
        out << ",\"transient_delta_before\":";
        emit_int_array(out, delta_before);
        out << ",\"transient_direct_after\":[],\"transient_delta_after\":[]"
            << ",\"E1_tree_hash\":" << epoch1.snapshot.hash
            << ",\"E1_leaf_stable_id_hash\":" << epoch1.exact_leaf_hash
            << ",\"E1_pivot_stable_id_hash\":" << epoch1.exact_pivot_hash
            << ",\"E1_leaf_set_exact\":true,\"E1_pivot_set_subset_of_live\":true}\n";
        out << "{\"record\":\"epoch\",\"epoch\":1,\"phase\":\"E1\",\"tree_hash\":"
            << epoch1.snapshot.hash << ",\"leaf_stable_id_hash\":" << epoch1.exact_leaf_hash
            << ",\"pivot_stable_id_hash\":" << epoch1.exact_pivot_hash
            << ",\"live_stable_id_hash\":" << epoch1.expected_live_hash
            << ",\"live_count\":" << epoch1.expected_live_ids.size()
            << ",\"leaf_set_exact\":true,\"pivot_set_subset_of_live\":true,\"pivot_stable_ids\":";
        emit_int_array(out, epoch1.exact_pivot_ids); out << "}\n";
      } else {
        die("unexpected trace opcode");
      }
      if (!out) die("failed writing result JSONL");
    }

    if (!have_e1 || state.rebuilds() != 1 || state.base_deletes() != 1 ||
        state.direct_inserts() != 2 || state.delta_inserts() != 1 ||
        state.capacity_rejects() != 1 || state.certificate_rejects() != 0 ||
        !state.direct_ids().empty() || !state.delta_ids().empty() || state.rebuild_required() ||
        state.active_count() != static_cast<int>(header.base_n) + 1) {
      die("G2 terminal state violates strict witness contract");
    }
    epoch1.assert_unchanged(runtime);
    out.close();

    int device = 0;
    cudaDeviceProp prop{};
    G2_CUDA(cudaGetDevice(&device));
    G2_CUDA(cudaGetDeviceProperties(&prop, device));
    std::ofstream summary(args.summary);
    if (!summary) die("cannot open G2 summary");
    summary << "{\n"
            << "  \"schema\": \"safe-c1-g2-rebuild-witness-v4\",\n"
            << "  \"status\": \"PASS_G2_REBUILD_WITNESS_PENDING_INDEPENDENT_VALIDATOR\",\n"
            << "  \"scope\": \"one strict stable-ID seeded immediate-rebuild witness; no performance/latency/general rebuild/range/C2/all-input tie claim\",\n"
            << "  \"archived_incremental_updater_used\": false,\n"
            << "  \"rebuild\": {\"trigger\": \"base_delete_immediate\", \"metadata_only_release\": true, \"full_pool_preserved\": true, \"active_set_preserved\": true, \"transient_tiers_cleared\": true},\n"
            << "  \"E0\": {\"tree_hash\": " << epoch0.snapshot.hash << ", \"leaf_stable_id_hash\": " << epoch0.exact_leaf_hash << ", \"pivot_stable_id_hash\": " << epoch0.exact_pivot_hash << ", \"leaf_set_exact\": true, \"pivot_set_subset_of_live\": true},\n"
            << "  \"E1\": {\"tree_hash\": " << epoch1.snapshot.hash << ", \"leaf_stable_id_hash\": " << epoch1.exact_leaf_hash << ", \"pivot_stable_id_hash\": " << epoch1.exact_pivot_hash << ", \"leaf_set_exact\": true, \"pivot_set_subset_of_live\": true},\n"
            << "  \"active_hash\": {\"before_rebuild\": " << before_active_hash << ", \"after_rebuild\": " << after_active_hash << "},\n"
            << "  \"immutable_full_pool_hash\": {\"host\": " << host_pool_hash << ", \"device_after_rebuild\": " << after_pool_hash << "},\n"
            << "  \"tie_witness\": {\"static_epoch_queries_checked\": " << ties.static_queries << ", \"dynamic_boundaries_checked\": " << ties.dynamic_boundaries << "},\n"
            << "  \"dynamic_query\": {\"method\": \"real GTS base top-k + visited-leaf sidecars + global delta; checked against independent full active-set oracle\", \"queries\": 3},\n"
            << "  \"gpu\": {\"name\": \"" << prop.name << "\", \"compute_capability\": \"" << prop.major << '.' << prop.minor << "\"},\n"
            << "  \"not_established\": [\"performance benefit\", \"latency\", \"general rebuild policy\", \"range-ID exporter\", \"C2 calibration/drift\", \"all-input tie semantics\"]\n"
            << "}\n";
    summary.close();

    G2_CUDA(cudaFree(candidate_ids_d)); candidate_ids_d = nullptr;
    G2_CUDA(cudaFree(candidate_distances_d)); candidate_distances_d = nullptr;
    G2_CUDA(cudaFree(query_d)); query_d = nullptr;
    release_tree_full(runtime);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    if (candidate_ids_d) cudaFree(candidate_ids_d);
    if (candidate_distances_d) cudaFree(candidate_distances_d);
    if (query_d) cudaFree(query_d);
    release_tree_full(runtime);
    return 2;
  }
}
