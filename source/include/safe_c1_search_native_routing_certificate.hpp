#pragma once

// CPU-only Safe-C1 routing certificate that exactly follows the child
// boundaries used by source_gts_incremental/include/search_v2.cuh.
//
// For a parent pivot p and consecutive child minima m_i:
//   non-last child i: m_i + eps < d(p,x) < m_(i+1) - eps
//   last child:         m_last + eps < d(p,x)
// Any boundary, missing sibling, overlap/non-uniqueness, non-finite value,
// inconsistent pivot, or incomplete path is rejected to global delta.
//
// Deliberately no max-distance field is used: search_v2 does not use it in
// nodeProcessKnn/nodeProcessRnn pruning.

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace safe_c1_routing {

struct Node {
  int pivot_id = -1;
  float min_distance = 0.0F;
  bool is_leaf = false;
  bool empty = true;
};

inline void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error("Safe-C1 sibling-boundary certificate: " + message);
  }
}

class SearchNativeCertificate {
 public:
  SearchNativeCertificate(std::vector<Node> nodes, int fanout, int tree_height,
                          float strict_epsilon = 1.0e-5F)
      : nodes_(std::move(nodes)),
        fanout_(fanout),
        tree_height_(tree_height),
        epsilon_(strict_epsilon) {
    require(!nodes_.empty() && fanout_ > 1 && tree_height_ > 1,
            "invalid frozen routing snapshot dimensions");
    require(std::isfinite(epsilon_) && epsilon_ > 0.0F,
            "invalid strict epsilon");
  }

  // The callback supplies d(inserted_object, pivot_id) from the immutable pool.
  template <class PivotDistance>
  int certify_leaf(PivotDistance&& distance_to_pivot) const {
    int parent = 0;
    for (int level = 0; level < tree_height_ - 1; ++level) {
      const int pivot = common_pivot_or_reject(parent);
      if (pivot < 0) return -1;
      const float distance = distance_to_pivot(pivot);
      if (!std::isfinite(distance) || distance < 0.0F) return -1;
      const int child = only_strict_child(parent, distance);
      if (child < 0) return -1;
      if (nodes_[static_cast<std::size_t>(child)].is_leaf) return child;
      parent = child;
    }
    return -1;
  }

  // Matches search_v2's min and next-sibling-min intervals. The final sibling
  // intentionally has no max-distance upper bound because search_v2 has none.
  [[nodiscard]] int only_strict_child(int parent, float distance) const {
    if (parent < 0 || parent >= static_cast<int>(nodes_.size()) ||
        !std::isfinite(distance) || distance < 0.0F ||
        common_pivot_or_reject(parent) < 0) {
      return -1;
    }
    int match = -1;
    for (int slot = 0; slot < fanout_; ++slot) {
      const int child = child_id(parent, slot);
      if (!valid_nonempty(child)) continue;
      const Node& node = nodes_[static_cast<std::size_t>(child)];
      if (!std::isfinite(node.min_distance) ||
          !(distance > node.min_distance + epsilon_)) {
        continue;
      }

      bool upper_ok = true;
      if (slot + 1 < fanout_) {
        const int next = child_id(parent, slot + 1);
        // search_v2 dereferences the physical next sibling. A missing or empty
        // boundary cannot be certified safely, so fail closed.
        if (!valid_nonempty(next)) return -1;
        const float next_min = nodes_[static_cast<std::size_t>(next)].min_distance;
        if (!std::isfinite(next_min)) return -1;
        upper_ok = distance < next_min - epsilon_;
      }
      if (!upper_ok) continue;
      if (match != -1) return -1;
      match = child;
    }
    return match;
  }

 private:
  [[nodiscard]] int child_id(int parent, int slot) const {
    return parent * fanout_ + slot + 1;
  }

  [[nodiscard]] bool valid_nonempty(int node_id) const {
    return node_id >= 0 && node_id < static_cast<int>(nodes_.size()) &&
           !nodes_[static_cast<std::size_t>(node_id)].empty;
  }

  [[nodiscard]] int common_pivot_or_reject(int parent) const {
    int pivot = -1;
    float previous_min = 0.0F;
    for (int slot = 0; slot < fanout_; ++slot) {
      const int child = child_id(parent, slot);
      // A split parent in this GTS layout has a physical full sibling block.
      // Anything else cannot safely mirror search_v2's child+1 dereference.
      if (!valid_nonempty(child)) return -1;
      const Node& node = nodes_[static_cast<std::size_t>(child)];
      if (node.pivot_id < 0 || !std::isfinite(node.min_distance)) return -1;
      if (pivot == -1) pivot = node.pivot_id;
      if (pivot != node.pivot_id) return -1;
      if (slot > 0 && !(node.min_distance > previous_min + epsilon_)) {
        return -1;  // non-monotonic/equal sibling minima: no unique annulus proof
      }
      previous_min = node.min_distance;
    }
    return pivot;
  }

  std::vector<Node> nodes_;
  int fanout_ = 0;
  int tree_height_ = 0;
  float epsilon_ = 0.0F;
};

}  // namespace safe_c1_routing
