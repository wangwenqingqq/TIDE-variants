#pragma once

// CPU-only tie contract for the limited Safe-C1 G1 witness domain.
//
// The copied base GTS thrust sort does not include stable ID in its key, so it
// cannot establish a canonical (distance, stable_id) result when a base top-k
// boundary is tied. This helper deliberately does NOT solve global tie
// semantics. It permits a run only when:
//   (1) for every static base query, the first min(k+1, base_n) exact
//       squared-L2 keys AND modeled GTS float-L2 sort keys are pairwise
//       distinct, and their first-k ID order agrees; ties strictly after that
//       prefix are permitted;
//   (2) at every dynamic trace kNN state, ranks k and k+1 have distinct exact
//       quantized squared-L2 keys (when k+1 exists).
// Any violation throws with ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN. The caller
// must not emit a PASS result after that exception.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace safe_c1_tie {

struct WitnessAudit {
  int static_base_queries_checked = 0;
  int dynamic_k_boundary_queries_checked = 0;
};

class TieFreeWitnessChecker {
 public:
  TieFreeWitnessChecker(const std::vector<std::int16_t>& pool,
                        const std::vector<std::int16_t>& queries,
                        int dimension, int base_count, int k)
      : pool_(pool), queries_(queries), dimension_(dimension),
        base_count_(base_count), k_(k) {
    if (dimension_ <= 0 || base_count_ <= 0 || k_ <= 0 || k_ > base_count_ ||
        pool_.size() % static_cast<std::size_t>(dimension_) != 0 ||
        queries_.size() % static_cast<std::size_t>(dimension_) != 0 ||
        base_count_ > pool_size()) {
      throw std::runtime_error("Safe-C1 tie witness: invalid quantized input dimensions");
    }
  }

  [[nodiscard]] int pool_size() const {
    return static_cast<int>(pool_.size() / static_cast<std::size_t>(dimension_));
  }

  [[nodiscard]] int query_size() const {
    return static_cast<int>(queries_.size() / static_cast<std::size_t>(dimension_));
  }

  // Static GTS top-k has no StableId key. We only need its returned prefix
  // to be deterministic: check the first min(k+1, base_n) values under both
  // the exact quantized oracle and the modeled legacy float key, then require
  // their first-k ID orders to agree. Ties strictly after this prefix cannot
  // affect static top-k membership and are intentionally allowed.
  void require_static_base_top_k_plus_one_tie_free(WitnessAudit* audit) const {
    for (int query_id = 0; query_id < query_size(); ++query_id) {
      std::vector<std::pair<std::uint64_t, int>> exact_keys;
      std::vector<std::pair<float, int>> gts_float_keys;
      exact_keys.reserve(static_cast<std::size_t>(base_count_));
      gts_float_keys.reserve(static_cast<std::size_t>(base_count_));
      for (int id = 0; id < base_count_; ++id) {
        exact_keys.emplace_back(squared_l2_key(id, query_id), id);
        gts_float_keys.emplace_back(modeled_gts_float_l2_sort_key(id, query_id), id);
      }
      std::sort(exact_keys.begin(), exact_keys.end());
      std::sort(gts_float_keys.begin(), gts_float_keys.end());
      const int prefix_count = std::min(base_count_, k_ + 1);
      for (int rank = 1; rank < prefix_count; ++rank) {
        if (exact_keys[static_cast<std::size_t>(rank - 1)].first ==
            exact_keys[static_cast<std::size_t>(rank)].first) {
          abort_unproved("static_base_top_k_plus_one_exact_squared_l2_tie q=" +
                         std::to_string(query_id) + " ranks=" +
                         std::to_string(rank - 1) + "," + std::to_string(rank) +
                         " stable_ids=" +
                         std::to_string(exact_keys[static_cast<std::size_t>(rank - 1)].second) + "," +
                         std::to_string(exact_keys[static_cast<std::size_t>(rank)].second) +
                         " squared_l2=" +
                         std::to_string(exact_keys[static_cast<std::size_t>(rank)].first));
        }
        if (gts_float_keys[static_cast<std::size_t>(rank - 1)].first ==
            gts_float_keys[static_cast<std::size_t>(rank)].first) {
          abort_unproved("static_base_top_k_plus_one_modeled_gts_float_key_tie q=" +
                         std::to_string(query_id) + " ranks=" +
                         std::to_string(rank - 1) + "," + std::to_string(rank) +
                         " stable_ids=" +
                         std::to_string(gts_float_keys[static_cast<std::size_t>(rank - 1)].second) + "," +
                         std::to_string(gts_float_keys[static_cast<std::size_t>(rank)].second) +
                         " modeled_float_l2=" +
                         std::to_string(gts_float_keys[static_cast<std::size_t>(rank)].first));
        }
      }
      for (int rank = 0; rank < std::min(k_, base_count_); ++rank) {
        if (exact_keys[static_cast<std::size_t>(rank)].second !=
            gts_float_keys[static_cast<std::size_t>(rank)].second) {
          abort_unproved("static_base_top_k_order_mismatch_exact_vs_modeled_gts_float q=" +
                         std::to_string(query_id) + " rank=" + std::to_string(rank) +
                         " exact_id=" +
                         std::to_string(exact_keys[static_cast<std::size_t>(rank)].second) +
                         " modeled_gts_id=" +
                         std::to_string(gts_float_keys[static_cast<std::size_t>(rank)].second));
        }
      }
      if (audit != nullptr) ++audit->static_base_queries_checked;
    }
  }

  // Only the dynamic k boundary is required to be distinct. Ties wholly above
  // that boundary do not alter top-k membership and are not a claim of complete
  // tie semantics; final exact host ordering may still use StableId there.
  void require_dynamic_k_boundary_tie_free(
      const std::vector<std::uint8_t>& active, int query_id,
      WitnessAudit* audit) const {
    if (static_cast<int>(active.size()) != pool_size()) {
      throw std::runtime_error("Safe-C1 tie witness: active-set size mismatch");
    }
    if (query_id < 0 || query_id >= query_size()) {
      throw std::runtime_error("Safe-C1 tie witness: query ID outside witness pool");
    }
    std::vector<std::pair<std::uint64_t, int>> keys;
    keys.reserve(active.size());
    for (int id = 0; id < pool_size(); ++id) {
      if (active[static_cast<std::size_t>(id)] != 0) {
        keys.emplace_back(squared_l2_key(id, query_id), id);
      }
    }
    if (static_cast<int>(keys.size()) < k_) {
      throw std::runtime_error("Safe-C1 tie witness: active set below requested k");
    }
    std::sort(keys.begin(), keys.end());
    if (static_cast<int>(keys.size()) > k_ &&
        keys[static_cast<std::size_t>(k_ - 1)].first ==
            keys[static_cast<std::size_t>(k_)].first) {
      abort_unproved("dynamic_k_boundary_distance_tie q=" +
                     std::to_string(query_id) + " k=" + std::to_string(k_) +
                     " stable_ids=" +
                     std::to_string(keys[static_cast<std::size_t>(k_ - 1)].second) + "," +
                     std::to_string(keys[static_cast<std::size_t>(k_)].second) +
                     " squared_l2=" +
                     std::to_string(keys[static_cast<std::size_t>(k_ - 1)].first));
    }
    if (audit != nullptr) ++audit->dynamic_k_boundary_queries_checked;
  }

  // These two read-only accessors let the independent static GTS probe compare
  // returned IDs against the same exact/modelled keys used by this preflight.
  [[nodiscard]] std::uint64_t exact_squared_l2_key_for(
      int stable_id, int query_id) const {
    return squared_l2_key(stable_id, query_id);
  }

  [[nodiscard]] float modeled_gts_float_l2_key_for(
      int stable_id, int query_id) const {
    return modeled_gts_float_l2_sort_key(stable_id, query_id);
  }

 private:
  [[nodiscard]] std::uint64_t squared_l2_key(int stable_id, int query_id) const {
    if (stable_id < 0 || stable_id >= pool_size() || query_id < 0 ||
        query_id >= query_size()) {
      throw std::runtime_error("Safe-C1 tie witness: distance key ID out of range");
    }
    const std::size_t point_offset =
        static_cast<std::size_t>(stable_id) * static_cast<std::size_t>(dimension_);
    const std::size_t query_offset =
        static_cast<std::size_t>(query_id) * static_cast<std::size_t>(dimension_);
    std::uint64_t total = 0;
    for (int d = 0; d < dimension_; ++d) {
      const std::int64_t delta =
          static_cast<std::int64_t>(pool_[point_offset + static_cast<std::size_t>(d)]) -
          static_cast<std::int64_t>(queries_[query_offset + static_cast<std::size_t>(d)]);
      const std::uint64_t term = static_cast<std::uint64_t>(delta * delta);
      if (total > std::numeric_limits<std::uint64_t>::max() - term) {
        throw std::runtime_error("Safe-C1 tie witness: squared-L2 key overflow");
      }
      total += term;
    }
    return total;
  }

  [[nodiscard]] float modeled_gts_float_l2_sort_key(int stable_id,
                                                       int query_id) const {
    if (stable_id < 0 || stable_id >= pool_size() || query_id < 0 ||
        query_id >= query_size()) {
      throw std::runtime_error("Safe-C1 tie witness: GTS sort-key ID out of range");
    }
    const std::size_t point_offset =
        static_cast<std::size_t>(stable_id) * static_cast<std::size_t>(dimension_);
    const std::size_t query_offset =
        static_cast<std::size_t>(query_id) * static_cast<std::size_t>(dimension_);
    double accumulated = 0.0;
    for (int d = 0; d < dimension_; ++d) {
      const float diff =
          static_cast<float>(pool_[point_offset + static_cast<std::size_t>(d)]) -
          static_cast<float>(queries_[query_offset + static_cast<std::size_t>(d)]);
      accumulated += static_cast<double>(diff * diff);
    }
    // dataProcessKnnVec stores sqrtf(result) into a double key payload.
    return std::sqrt(static_cast<float>(accumulated));
  }

  [[noreturn]] static void abort_unproved(const std::string& detail) {
    throw std::runtime_error(
        "ABORT_UNPROVED_BOUNDED_TIE_WITNESS_DOMAIN: " + detail);
  }

  const std::vector<std::int16_t>& pool_;
  const std::vector<std::int16_t>& queries_;
  int dimension_ = 0;
  int base_count_ = 0;
  int k_ = 0;
};

}  // namespace safe_c1_tie
