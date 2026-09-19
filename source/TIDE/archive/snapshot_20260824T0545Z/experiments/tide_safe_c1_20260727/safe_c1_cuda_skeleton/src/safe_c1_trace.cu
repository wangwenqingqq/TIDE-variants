// Minimal Safe-C1 CUDA/GTS trace harness skeleton.
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
#include "search_v2.cuh"

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
    hash = compute_hash();
  }

  void assert_unchanged(const BaseTreeRuntime& runtime) const {
    FrozenTreeSnapshot current;
    current.capture(runtime);
    if (current.hash != hash) fail("base tree geometry changed outside rebuild");
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

// Existing static GTS top-k path can be called only as a frozen base-tree probe.
// It does not include Safe-C1 sidecars/delta and must never be reported as the
// dynamic result. `query_vectors` is a managed/device buffer with qnum*dim floats.
TopKPairs probe_static_base_topk(BaseTreeRuntime& runtime, float* query_vectors,
                                int qnum, int k) {
  if (!runtime.ready() || query_vectors == nullptr || qnum <= 0 || k <= 0) {
    fail("invalid static top-k probe arguments");
  }
  int* result_ids = nullptr;
  CHECK(cudaMallocManaged((void**)&result_ids, static_cast<std::size_t>(qnum) * k * sizeof(int)));
  update_disk = false;
  searchIndexKnnV2(runtime.data_d, runtime.node_list, runtime.id_list,
                    runtime.max_node_num, query_vectors, result_ids, qnum, k,
                    runtime.tree_height, runtime.data_info, runtime.empty_list,
                    runtime.data_s, runtime.size_s);
  CHECK(cudaDeviceSynchronize());

  TopKPairs output;
  output.ids.assign(result_ids, result_ids + static_cast<std::size_t>(qnum) * k);
  output.distances.assign(res_dis, res_dis + static_cast<std::size_t>(qnum) * k);
  CHECK(cudaFree(result_ids));
  CHECK(cudaFree(res_dis));
  res_dis = nullptr;
  return output;
}

// Required implementation seam.  The historical static range API reports only
// a count (`res`) and therefore cannot support exact stable-ID range auditing.
// A new active-filtered base leaf scan + sidecar leaf scan + global delta scan
// must be installed here before any E1-G trace is executed.
struct SafeQueryApi {
  [[nodiscard]] static const char* range_status() {
    return "TODO: export stable-ID range pairs from frozen base leaves + sidecar + delta";
  }
  [[nodiscard]] static const char* topk_status() {
    return "TODO: merge frozen base top-k candidates + sidecar + delta with (distance,stable_id) tie-break";
  }
};

}  // namespace safe_c1

int main(int argc, char** argv) {
  // This command intentionally avoids any CUDA context/workload.  It exists so
  // a build artifact advertises the contract; actual E1-G execution requires
  // trace loader, custom query kernels, and an external oracle invocation.
  if (argc == 2 && std::string(argv[1]) == "--skeleton-info") {
    std::cout << "{\"target\":\"GTS_safe_c1_trace\","
              << "\"legacy_incremental_insert\":false,"
              << "\"base_only_builder\":true,"
              << "\"frozen_snapshot\":true,"
              << "\"stable_sidecar_delta\":true,"
              << "\"static_topk_probe\":true,"
              << "\"safe_range_merge\":\"TODO\","
              << "\"safe_topk_merge\":\"TODO\"}\n";
    return 0;
  }
  std::cerr << "GTS_safe_c1_trace is a no-GPU-execution skeleton. "
            << "Use --skeleton-info; do not treat it as an E1-G result runner.\n";
  return argc == 1 ? 0 : 2;
}
