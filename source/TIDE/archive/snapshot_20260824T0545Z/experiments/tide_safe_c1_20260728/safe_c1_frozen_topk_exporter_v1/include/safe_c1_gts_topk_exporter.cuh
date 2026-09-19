#pragma once

// Isolated Safe-C1 top-k exporter.  It never includes the archived mutable
// incremental_insert/update headers.  The frozen GTS base is queried through
// searchIndexKnnV2; strict-contained sidecars and global delta are external,
// exactly scanned, and then canonically merged.

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "tree.cuh"
#include "safe_c1_search_v2_leaf_export.cuh"
#include "safe_c1_merge_contract.hpp"

namespace safe_c1_exporter {

inline void cuda_require(cudaError_t status, const char* where) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string("Safe-C1 CUDA failure at ") + where +
                             ": " + cudaGetErrorString(status));
  }
}

inline std::uint64_t fnv1a_append(std::uint64_t seed, const void* address,
                                  std::size_t bytes) {
  const auto* data = static_cast<const unsigned char*>(address);
  std::uint64_t result = seed;
  for (std::size_t index = 0; index < bytes; ++index) {
    result ^= static_cast<std::uint64_t>(data[index]);
    result *= 1099511628211ULL;
  }
  return result;
}

template <class T>
inline std::uint64_t hash_value(std::uint64_t seed, const T& value) {
  return fnv1a_append(seed, &value, sizeof(T));
}

// config.cuh maps short to float in this archived source, so use float here.
struct FrozenGtsBase {
  float* data_d = nullptr;
  TN* node_list = nullptr;
  int* id_list = nullptr;
  int* max_node_num = nullptr;
  int* data_info = nullptr;
  int* empty_list = nullptr;
  char* data_s = nullptr;
  int* size_s = nullptr;
  float* max_distance = nullptr;  // archived max_dis_d, for certificate integrity
  int tree_height = 0;
  int base_count = 0;

  void validate() const {
    require(data_d != nullptr && node_list != nullptr && id_list != nullptr &&
                max_node_num != nullptr && data_info != nullptr && empty_list != nullptr,
            "frozen GTS base has a null runtime pointer");
    require(tree_height > 0 && base_count > 0 && data_info[1] == base_count,
            "frozen GTS base has inconsistent object count");
    require(max_node_num[0] > 0, "frozen GTS base reports no nodes");
  }
};

// Hash initialized semantic fields, not TN padding or uninitialized empty nodes.
struct FrozenBaseFingerprint {
  std::uint64_t hash = 0;
  int logical_leaf_ids = 0;
  int nonempty_nodes = 0;

  static FrozenBaseFingerprint capture(const FrozenGtsBase& base) {
    base.validate();
    cuda_require(cudaDeviceSynchronize(), "fingerprint sync");
    const int node_count = base.max_node_num[0];
    std::vector<TN> nodes(static_cast<std::size_t>(node_count));
    std::vector<int> empty(static_cast<std::size_t>(node_count));
    std::vector<float> max_distance;
    cuda_require(cudaMemcpy(nodes.data(), base.node_list, nodes.size() * sizeof(TN),
                            cudaMemcpyDeviceToHost),
                 "fingerprint node copy");
    cuda_require(cudaMemcpy(empty.data(), base.empty_list, empty.size() * sizeof(int),
                            cudaMemcpyDeviceToHost),
                 "fingerprint empty copy");
    if (base.max_distance != nullptr) {
      max_distance.resize(static_cast<std::size_t>(node_count));
      cuda_require(cudaMemcpy(max_distance.data(), base.max_distance,
                              max_distance.size() * sizeof(float),
                              cudaMemcpyDeviceToHost),
                   "fingerprint max-distance copy");
    }

    FrozenBaseFingerprint result;
    std::vector<std::uint8_t> seen_base(static_cast<std::size_t>(base.base_count), 0);
    std::uint64_t value = 1469598103934665603ULL;
    value = hash_value(value, base.tree_height);
    value = hash_value(value, base.base_count);
    value = hash_value(value, node_count);
    const int fanout = TREE_ORDER;
    value = hash_value(value, fanout);
    for (int node_id = 0; node_id < node_count; ++node_id) {
      value = hash_value(value, node_id);
      value = hash_value(value, empty[static_cast<std::size_t>(node_id)]);
      if (empty[static_cast<std::size_t>(node_id)] != 0) continue;
      const TN& node = nodes[static_cast<std::size_t>(node_id)];
      value = hash_value(value, node.pid);
      value = hash_value(value, node.min_dis);
      value = hash_value(value, node.size);
      value = hash_value(value, node.lid);
      value = hash_value(value, node.is_leaf);
      if (!max_distance.empty()) {
        value = hash_value(value, max_distance[static_cast<std::size_t>(node_id)]);
      }
      ++result.nonempty_nodes;
      if (node.is_leaf != 1) continue;
      require(node.lid >= 0 && node.size >= 0, "invalid frozen leaf layout");
      std::vector<int> ids(static_cast<std::size_t>(node.size));
      if (!ids.empty()) {
        cuda_require(cudaMemcpy(ids.data(), base.id_list + node.lid,
                                ids.size() * sizeof(int), cudaMemcpyDeviceToHost),
                     "fingerprint leaf ID copy");
        for (int id : ids) {
          require(id >= 0 && id < base.base_count,
                  "frozen GTS leaf contains a non-base stable ID");
          require(seen_base[static_cast<std::size_t>(id)] == 0,
                  "frozen GTS leaf payload contains duplicate stable ID");
          seen_base[static_cast<std::size_t>(id)] = 1;
          value = hash_value(value, id);
        }
      }
      result.logical_leaf_ids += node.size;
    }
    require(result.logical_leaf_ids == base.base_count,
            "frozen GTS leaves do not cover exactly the base stable IDs");
    result.hash = value;
    return result;
  }

  void assert_matches(const FrozenGtsBase& base) const {
    const FrozenBaseFingerprint current = capture(base);
    require(current.hash == hash &&
                current.logical_leaf_ids == logical_leaf_ids &&
                current.nonempty_nodes == nonempty_nodes,
            "frozen GTS base mutated outside explicit rebuild");
  }
};

// This adapter never enables the legacy learned residual/C2 path.  Call this
// before constructing the static-base oracle and record that call in its gate.
inline void set_frozen_base_residual_mode_zero() {
  const int mode = 0;
  cuda_require(cudaMemcpyToSymbol(c_rp_mode, &mode, sizeof(mode), 0,
                                  cudaMemcpyHostToDevice),
               "set frozen-base residual mode=0");
}

struct FullVectorPoolDevice {
  const float* vectors = nullptr;  // device/managed full immutable stable-ID pool
  int pool_count = 0;
  int dimension = 0;

  void validate() const {
    require(vectors != nullptr && pool_count > 0 && dimension > 0,
            "full immutable vector pool is not initialized");
  }
};

// A dynamic query is blocked until these two independent conditions are true.
struct StaticBaseTopKGate {
  bool base_oracle_passed = false;
  bool residual_mode_zero_verified = false;
  bool visited_leaf_export_verified = false;
  bool strict_sidecar_visibility_verified = false;
  std::string evidence_label;

  void require_ready() const {
    require(base_oracle_passed,
            "base traversal oracle gate has not passed");
    require(residual_mode_zero_verified,
            "residual pruning mode=0 has not been verified");
    require(visited_leaf_export_verified,
            "GTS visited-leaf exporter has not passed its audit gate");
    require(strict_sidecar_visibility_verified,
            "strict sidecar visibility/certificate gate has not passed");
    require(!evidence_label.empty(),
            "base traversal oracle gate has no provenance label");
  }
};

// Base deletion must never be hidden by the frozen base query.  The calling
// update layer marks the barrier; only publish_rebuild clears it.
class BaseDeleteBarrier {
 public:
  void mark_base_delete() { pending_ = true; }
  void publish_rebuild() { pending_ = false; }
  [[nodiscard]] bool pending() const { return pending_; }
  void require_queryable() const {
    require(!pending_,
            "base deletion is pending; rebuild must publish a fresh frozen GTS base");
  }

 private:
  bool pending_ = false;
};

// Recompute all selected candidate distances in one canonical float64 arithmetic.
// Legacy GTS float res_dis is freed and never compared with delta distances.
__global__ void canonical_l2_for_ids(const float* full_pool, const float* query,
                                     const int* ids, double* distances,
                                     int count, int dimension) {
  const int offset = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (offset >= count) return;
  const int stable_id = ids[offset];
  const float* vector = full_pool +
      static_cast<std::size_t>(stable_id) * static_cast<std::size_t>(dimension);
  double sum = 0.0;
  for (int axis = 0; axis < dimension; ++axis) {
    const double difference = static_cast<double>(vector[axis]) -
                              static_cast<double>(query[axis]);
    sum += difference * difference;
  }
  distances[offset] = sqrt(sum);
}

struct FrozenTraversalResult {
  std::vector<StableId> base_topk_ids;
  std::vector<int> visited_leaf_ids;  // deduplicated actual mergeLNodeKnn leaves
};

struct DynamicTopKResult {
  std::vector<Candidate> ranked;
  std::vector<StableId> base_candidates_from_gts;
  std::vector<int> visited_leaf_ids;
  std::size_t visited_sidecar_candidates = 0;
  std::size_t delta_candidates = 0;
  std::uint64_t frozen_hash = 0;
};

class FrozenTopKExporter {
 public:
  FrozenTopKExporter(FrozenGtsBase base, StaticBaseTopKGate gate)
      : base_(base), gate_(std::move(gate)),
        fingerprint_(FrozenBaseFingerprint::capture(base_)) {
    gate_.require_ready();
  }

  void mark_base_delete() { base_delete_barrier_.mark_base_delete(); }

  // Only after logged rebuild plus a new independent base-only oracle gate.
  void publish_rebuild(FrozenGtsBase rebuilt_base, StaticBaseTopKGate rebuilt_gate) {
    rebuilt_base.validate();
    rebuilt_gate.require_ready();
    base_ = rebuilt_base;
    gate_ = std::move(rebuilt_gate);
    fingerprint_ = FrozenBaseFingerprint::capture(base_);
    base_delete_barrier_.publish_rebuild();
  }

  [[nodiscard]] const FrozenBaseFingerprint& fingerprint() const {
    return fingerprint_;
  }

  // query points to one device/managed vector. direct sidecars are stored
  // by frozen leaf; only buckets whose IDs were exported by the same GTS
  // traversal are gathered. Global delta remains the exact fallback tier.
  DynamicTopKResult query_one(const float* query,
                              const FullVectorPoolDevice& full_pool,
                              const ExtraTierIds& extras, int k) {
    base_.validate();
    full_pool.validate();
    gate_.require_ready();
    base_delete_barrier_.require_queryable();
    require(query != nullptr && k > 0 && k <= base_.base_count,
            "invalid frozen-base top-k query argument");
    require(full_pool.pool_count >= base_.base_count &&
                full_pool.dimension == base_.data_info[0],
            "full immutable pool disagrees with frozen GTS base schema");
    fingerprint_.assert_matches(base_);

    FrozenTraversalResult traversal = run_frozen_gts_topk(query, k);
    validate_base_topk_ids(traversal.base_topk_ids, k, base_.base_count);
    require(!traversal.visited_leaf_ids.empty(),
            "frozen GTS top-k traversal exported no leaf");
    const SelectedExtraIds selected = extras.select_for_visited_leaves(
        traversal.visited_leaf_ids, base_.base_count, full_pool.pool_count);
    const std::vector<StableId> extra_ids = selected.combined();

    std::vector<StableId> all_ids;
    all_ids.reserve(traversal.base_topk_ids.size() + extra_ids.size());
    all_ids.insert(all_ids.end(), traversal.base_topk_ids.begin(),
                   traversal.base_topk_ids.end());
    all_ids.insert(all_ids.end(), extra_ids.begin(), extra_ids.end());
    const std::vector<double> all_distances =
        canonical_distances(query, full_pool, all_ids);
    const std::size_t base_size = traversal.base_topk_ids.size();
    std::vector<double> base_distances(all_distances.begin(),
                                       all_distances.begin() + base_size);
    std::vector<double> extra_distances(all_distances.begin() + base_size,
                                        all_distances.end());

    DynamicTopKResult result;
    result.ranked = merge_canonical_topk(
        traversal.base_topk_ids, base_distances, extra_ids, extra_distances, k,
        base_.base_count, full_pool.pool_count);
    result.base_candidates_from_gts = std::move(traversal.base_topk_ids);
    result.visited_leaf_ids = std::move(traversal.visited_leaf_ids);
    result.visited_sidecar_candidates = selected.visited_sidecar_ids.size();
    result.delta_candidates = selected.delta_ids.size();
    result.frozen_hash = fingerprint_.hash;
    fingerprint_.assert_matches(base_);
    return result;
  }

 private:
  FrozenTraversalResult run_frozen_gts_topk(const float* query, int k) {
    // The copied isolated search_v2 exporter owns process-wide
    // st/update_disk/res_dis and its optional leaf-export hook. This adapter is
    // intentionally single-query/single-process.
    while (!st.empty()) st.pop();
    update_disk = false;
    require(res_dis == nullptr,
            "legacy static-search distance buffer is still live");
    require(safe_c1_leaf_export == nullptr &&
                safe_c1_leaf_export_count == nullptr &&
                safe_c1_leaf_export_overflow == nullptr,
            "visited-leaf hook is live from another caller");

    int* raw_ids = nullptr;
    int* raw_leaves = nullptr;
    int* leaf_count = nullptr;
    int* leaf_overflow = nullptr;
    const int leaf_capacity = base_.max_node_num[0];
    cuda_require(cudaMallocManaged(reinterpret_cast<void**>(&raw_ids),
                                   static_cast<std::size_t>(k) * sizeof(int)),
                 "allocate frozen base top-k IDs");
    try {
      cuda_require(cudaMallocManaged(reinterpret_cast<void**>(&raw_leaves),
                                     static_cast<std::size_t>(leaf_capacity) * sizeof(int)),
                   "allocate visited-leaf export");
      cuda_require(cudaMallocManaged(reinterpret_cast<void**>(&leaf_count), sizeof(int)),
                   "allocate visited-leaf count");
      cuda_require(cudaMallocManaged(reinterpret_cast<void**>(&leaf_overflow), sizeof(int)),
                   "allocate visited-leaf overflow flag");
      leaf_count[0] = 0;
      leaf_overflow[0] = 0;
      safe_c1_leaf_export = raw_leaves;
      safe_c1_leaf_export_count = leaf_count;
      safe_c1_leaf_export_capacity = leaf_capacity;
      safe_c1_leaf_export_overflow = leaf_overflow;

      searchIndexKnnV2(base_.data_d, base_.node_list, base_.id_list,
                        base_.max_node_num, const_cast<float*>(query), raw_ids,
                        /*qnum=*/1, k, base_.tree_height, base_.data_info,
                        base_.empty_list, base_.data_s, base_.size_s);
      cuda_require(cudaDeviceSynchronize(), "frozen GTS top-k sync");
      cuda_require(cudaGetLastError(), "frozen GTS top-k status");
      require(leaf_overflow[0] == 0 && leaf_count[0] >= 0 &&
                  leaf_count[0] <= leaf_capacity,
              "visited-leaf export overflow");
      FrozenTraversalResult result;
      result.base_topk_ids.assign(raw_ids, raw_ids + k);
      result.visited_leaf_ids.assign(raw_leaves, raw_leaves + leaf_count[0]);
      std::sort(result.visited_leaf_ids.begin(), result.visited_leaf_ids.end());
      result.visited_leaf_ids.erase(
          std::unique(result.visited_leaf_ids.begin(), result.visited_leaf_ids.end()),
          result.visited_leaf_ids.end());
      validate_exported_leaves(result.visited_leaf_ids);

      clear_leaf_export_hook();
      cuda_require(cudaFree(raw_ids), "free frozen base top-k IDs");
      cuda_require(cudaFree(raw_leaves), "free visited-leaf export");
      cuda_require(cudaFree(leaf_count), "free visited-leaf count");
      cuda_require(cudaFree(leaf_overflow), "free visited-leaf overflow flag");
      raw_ids = raw_leaves = leaf_count = leaf_overflow = nullptr;
      if (res_dis != nullptr) {
        cuda_require(cudaFree(res_dis), "free legacy GTS reported distances");
        res_dis = nullptr;
      }
      return result;
    } catch (...) {
      clear_leaf_export_hook();
      if (raw_ids != nullptr) cudaFree(raw_ids);
      if (raw_leaves != nullptr) cudaFree(raw_leaves);
      if (leaf_count != nullptr) cudaFree(leaf_count);
      if (leaf_overflow != nullptr) cudaFree(leaf_overflow);
      if (res_dis != nullptr) {
        cudaFree(res_dis);
        res_dis = nullptr;
      }
      throw;
    }
  }

  static void clear_leaf_export_hook() {
    safe_c1_leaf_export = nullptr;
    safe_c1_leaf_export_count = nullptr;
    safe_c1_leaf_export_capacity = 0;
    safe_c1_leaf_export_overflow = nullptr;
  }

  void validate_exported_leaves(const std::vector<int>& leaves) const {
    require(!leaves.empty(), "frozen GTS traversal produced an empty leaf set");
    for (int leaf : leaves) {
      require(leaf >= 0 && leaf < base_.max_node_num[0],
              "visited-leaf exporter returned out-of-range node");
      TN node{};
      int empty = 1;
      cuda_require(cudaMemcpy(&node, base_.node_list + leaf, sizeof(TN),
                              cudaMemcpyDeviceToHost),
                   "validate exported leaf node");
      cuda_require(cudaMemcpy(&empty, base_.empty_list + leaf, sizeof(int),
                              cudaMemcpyDeviceToHost),
                   "validate exported leaf empty flag");
      require(empty == 0 && node.is_leaf == 1,
              "visited-leaf exporter returned non-leaf node");
    }
  }

  static std::vector<double> canonical_distances(
      const float* query, const FullVectorPoolDevice& full_pool,
      const std::vector<StableId>& ids) {
    if (ids.empty()) return {};
    int* ids_d = nullptr;
    double* distances_d = nullptr;
    try {
      cuda_require(cudaMalloc(reinterpret_cast<void**>(&ids_d),
                              ids.size() * sizeof(int)),
                   "allocate canonical candidate IDs");
      cuda_require(cudaMalloc(reinterpret_cast<void**>(&distances_d),
                              ids.size() * sizeof(double)),
                   "allocate canonical candidate distances");
      cuda_require(cudaMemcpy(ids_d, ids.data(), ids.size() * sizeof(int),
                              cudaMemcpyHostToDevice),
                   "copy canonical candidate IDs");
      constexpr int threads = 256;
      const int blocks = static_cast<int>((ids.size() + threads - 1) / threads);
      canonical_l2_for_ids<<<blocks, threads>>>(
          full_pool.vectors, query, ids_d, distances_d,
          static_cast<int>(ids.size()), full_pool.dimension);
      cuda_require(cudaGetLastError(), "canonical candidate L2 launch");
      cuda_require(cudaDeviceSynchronize(), "canonical candidate L2 sync");
      std::vector<double> result(ids.size());
      cuda_require(cudaMemcpy(result.data(), distances_d,
                              result.size() * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "copy canonical candidate distances");
      cuda_require(cudaFree(ids_d), "free canonical candidate IDs");
      cuda_require(cudaFree(distances_d), "free canonical candidate distances");
      return result;
    } catch (...) {
      if (ids_d != nullptr) cudaFree(ids_d);
      if (distances_d != nullptr) cudaFree(distances_d);
      throw;
    }
  }

  FrozenGtsBase base_;
  StaticBaseTopKGate gate_;
  FrozenBaseFingerprint fingerprint_;
  BaseDeleteBarrier base_delete_barrier_;
};

}  // namespace safe_c1_exporter
