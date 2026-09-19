#pragma once

// Pure C++ part of the Safe-C1 dynamic top-k contract. It deliberately has no
// CUDA dependency so ranking, visited-leaf selection, and duplicate rules can
// be tested without a GPU.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace safe_c1_exporter {

using StableId = int;

struct Candidate {
  double distance = 0.0;
  StableId stable_id = -1;
};

inline bool candidate_less(const Candidate& left, const Candidate& right) {
  return left.distance != right.distance ? left.distance < right.distance
                                         : left.stable_id < right.stable_id;
}

inline void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error("Safe-C1 top-k contract: " + message);
}

inline void validate_candidate(const Candidate& candidate, int pool_count,
                               const char* tier) {
  require(candidate.stable_id >= 0 && candidate.stable_id < pool_count,
          std::string(tier) + " stable ID outside immutable pool");
  require(std::isfinite(candidate.distance) && candidate.distance >= 0.0,
          std::string(tier) + " has non-finite or negative canonical L2 distance");
}

// Direct objects are stored by their frozen certified leaf. Query code receives
// a leaf set exported by the actual frozen GTS traversal and fetches only those
// buckets. Global delta remains an exact fallback tier and is always scanned.
struct SelectedExtraIds {
  std::vector<StableId> visited_sidecar_ids;
  std::vector<StableId> delta_ids;

  [[nodiscard]] std::vector<StableId> combined() const {
    std::vector<StableId> result;
    result.reserve(visited_sidecar_ids.size() + delta_ids.size());
    result.insert(result.end(), visited_sidecar_ids.begin(), visited_sidecar_ids.end());
    result.insert(result.end(), delta_ids.begin(), delta_ids.end());
    return result;
  }
};

struct ExtraTierIds {
  std::unordered_map<int, std::vector<StableId>> sidecars_by_leaf;
  std::vector<StableId> delta_ids;

  // This is an update-time/state-construction invariant audit. It reads metadata
  // only; it does not cause distance computation for unvisited sidecars.
  void validate_storage(int base_count, int pool_count) const {
    require(base_count > 0 && base_count <= pool_count,
            "invalid base/pool counts while validating extra tiers");
    std::unordered_set<StableId> seen;
    const auto validate_ids = [&](const std::vector<StableId>& ids, const char* tier) {
      for (StableId id : ids) {
        require(id >= base_count && id < pool_count,
                std::string(tier) + " ID is not a reservoir stable ID");
        require(seen.insert(id).second,
                std::string("duplicate live stable ID across sidecar/delta tiers: ") +
                    std::to_string(id));
      }
    };
    for (const auto& bucket : sidecars_by_leaf) {
      require(bucket.first >= 0, "negative frozen leaf ID in sidecar map");
      validate_ids(bucket.second, "sidecar");
    }
    validate_ids(delta_ids, "delta");
  }

  [[nodiscard]] SelectedExtraIds select_for_visited_leaves(
      const std::vector<int>& visited_leaf_ids, int base_count, int pool_count) const {
    require(base_count > 0 && base_count <= pool_count,
            "invalid base/pool counts while selecting extra tiers");
    SelectedExtraIds selected;
    std::unordered_set<int> visited;
    std::unordered_set<StableId> seen;
    for (int leaf : visited_leaf_ids) {
      require(leaf >= 0, "negative leaf ID exported by frozen GTS traversal");
      if (!visited.insert(leaf).second) continue;
      const auto iterator = sidecars_by_leaf.find(leaf);
      if (iterator == sidecars_by_leaf.end()) continue;
      for (StableId id : iterator->second) {
        require(id >= base_count && id < pool_count,
                "visited sidecar ID is not a reservoir stable ID");
        require(seen.insert(id).second,
                "duplicate stable ID in selected visited sidecars");
        selected.visited_sidecar_ids.push_back(id);
      }
    }
    for (StableId id : delta_ids) {
      require(id >= base_count && id < pool_count,
              "delta ID is not a reservoir stable ID");
      require(seen.insert(id).second,
              "stable ID appears in selected sidecar and delta tiers");
      selected.delta_ids.push_back(id);
    }
    return selected;
  }
};

// The static base query has to return exactly k unique base IDs and must have
// independently passed a base-only oracle gate before this merge is enabled.
inline void validate_base_topk_ids(const std::vector<StableId>& ids, int k,
                                   int base_count) {
  require(k > 0 && base_count >= k, "base must contain at least k objects");
  require(static_cast<int>(ids.size()) == k,
          "frozen GTS static query did not return exactly k candidates");
  std::unordered_set<StableId> seen;
  for (StableId id : ids) {
    require(id >= 0 && id < base_count, "frozen GTS result contains non-base ID");
    require(seen.insert(id).second, "frozen GTS result contains duplicate stable ID");
  }
}

// Canonical ranking only consumes recomputed float64 L2 distances. Legacy GTS
// float res_dis is never compared with sidecar/delta distances.
inline std::vector<Candidate> merge_canonical_topk(
    const std::vector<StableId>& frozen_base_ids,
    const std::vector<double>& frozen_base_distances,
    const std::vector<StableId>& extra_ids,
    const std::vector<double>& extra_distances, int k, int base_count,
    int pool_count) {
  validate_base_topk_ids(frozen_base_ids, k, base_count);
  require(frozen_base_ids.size() == frozen_base_distances.size(),
          "base ID/distance cardinality mismatch");
  require(extra_ids.size() == extra_distances.size(),
          "extra ID/distance cardinality mismatch");

  std::vector<Candidate> candidates;
  candidates.reserve(frozen_base_ids.size() + extra_ids.size());
  std::unordered_set<StableId> seen;
  for (std::size_t i = 0; i < frozen_base_ids.size(); ++i) {
    Candidate candidate{frozen_base_distances[i], frozen_base_ids[i]};
    validate_candidate(candidate, pool_count, "base");
    require(seen.insert(candidate.stable_id).second, "duplicate base candidate");
    candidates.push_back(candidate);
  }
  for (std::size_t i = 0; i < extra_ids.size(); ++i) {
    Candidate candidate{extra_distances[i], extra_ids[i]};
    validate_candidate(candidate, pool_count, "extra");
    require(candidate.stable_id >= base_count,
            "extra tier contains a base stable ID; base deletion requires rebuild");
    require(seen.insert(candidate.stable_id).second,
            "stable ID appears in both frozen base and extra tier");
    candidates.push_back(candidate);
  }
  require(static_cast<int>(candidates.size()) >= k,
          "fewer than k active candidates after Safe-C1 update accounting");
  std::sort(candidates.begin(), candidates.end(), candidate_less);
  candidates.resize(k);
  return candidates;
}

}  // namespace safe_c1_exporter
