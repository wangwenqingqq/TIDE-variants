#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "safe_c1_search_native_routing_certificate.hpp"

namespace {

struct OldMinMaxNode {
  float min_distance = 0.0F;
  float max_distance = 0.0F;
  bool empty = true;
};

// Models the old FrozenTreeSnapshot rule: strict own [min,max], not next
// sibling minimum. It exists only to make the semantic counterexamples explicit.
int old_unique_minmax_child(const std::vector<OldMinMaxNode>& nodes,
                            int fanout, int parent, float distance,
                            float epsilon = 1.0e-5F) {
  int match = -1;
  for (int slot = 0; slot < fanout; ++slot) {
    const int child = parent * fanout + slot + 1;
    if (child < 0 || child >= static_cast<int>(nodes.size()) ||
        nodes[static_cast<std::size_t>(child)].empty) {
      continue;
    }
    const auto& node = nodes[static_cast<std::size_t>(child)];
    if (distance > node.min_distance + epsilon &&
        distance < node.max_distance - epsilon) {
      if (match != -1) return -1;
      match = child;
    }
  }
  return match;
}

void expect_equal(int actual, int expected, const std::string& label) {
  if (actual != expected) {
    throw std::runtime_error(label + ": expected " + std::to_string(expected) +
                             ", got " + std::to_string(actual));
  }
}

std::vector<safe_c1_routing::Node> root_two(float second_min) {
  std::vector<safe_c1_routing::Node> nodes(3);
  nodes[1] = safe_c1_routing::Node{101, 0.0F, true, false};
  nodes[2] = safe_c1_routing::Node{101, second_min, true, false};
  return nodes;
}

std::vector<OldMinMaxNode> old_root_two(float first_max, float second_min,
                                        float second_max) {
  std::vector<OldMinMaxNode> nodes(3);
  nodes[1] = OldMinMaxNode{0.0F, first_max, false};
  nodes[2] = OldMinMaxNode{second_min, second_max, false};
  return nodes;
}

}  // namespace

int main() {
  try {
    using safe_c1_routing::SearchNativeCertificate;

    // GAP: base max of child 1 can end before next sibling min. search_v2 still
    // uses child 2 min as child 1 upper boundary. Old max rule falls back;
    // search-native rule routes uniquely to child 1.
    {
      SearchNativeCertificate native(root_two(10.0F), 2, 2);
      const auto old = old_root_two(/*max1=*/5.0F, /*min2=*/10.0F, /*max2=*/20.0F);
      expect_equal(old_unique_minmax_child(old, 2, 0, 7.0F), -1,
                   "gap old min/max fallback");
      expect_equal(native.only_strict_child(0, 7.0F), 1,
                   "gap native sibling-boundary route");
      std::cout << "PASS gap_native_uses_next_sibling_min\n";
    }

    // STALE/OVERLAP BOUNDARY: old max1 crosses the next boundary. At r=m2 the
    // old own-max certificate accepts child 1, while the actual search boundary
    // is ambiguous and the new certificate rejects to delta.
    {
      SearchNativeCertificate native(root_two(10.0F), 2, 2);
      const auto old = old_root_two(/*max1 stale=*/15.0F, /*min2=*/10.0F,
                                    /*max2=*/20.0F);
      expect_equal(old_unique_minmax_child(old, 2, 0, 10.0F), 1,
                   "stale overlap old certificate accepts");
      expect_equal(native.only_strict_child(0, 10.0F), -1,
                   "stale overlap native rejects boundary");
      std::cout << "PASS stale_overlap_old_accepts_new_rejects\n";
    }

    // LAST SIBLING: search_v2 has no next-sibling upper term for the final
    // sibling. Old max-based routing rejects r>old_max; new rule accepts it.
    {
      SearchNativeCertificate native(root_two(10.0F), 2, 2);
      const auto old = old_root_two(/*max1=*/5.0F, /*min2=*/10.0F, /*max2=*/20.0F);
      expect_equal(old_unique_minmax_child(old, 2, 0, 25.0F), -1,
                   "last sibling old max fallback");
      expect_equal(native.only_strict_child(0, 25.0F), 2,
                   "last sibling native no fabricated upper bound");
      std::cout << "PASS last_sibling_no_max_upper\n";
    }

    // NORMAL BOUNDARY: both strict systems reject an exact sibling boundary,
    // but this makes the epsilon rule explicit.
    {
      SearchNativeCertificate native(root_two(10.0F), 2, 2);
      const auto old = old_root_two(/*max1=*/10.0F, /*min2=*/10.0F,
                                    /*max2=*/20.0F);
      expect_equal(old_unique_minmax_child(old, 2, 0, 10.0F), -1,
                   "normal boundary old reject");
      expect_equal(native.only_strict_child(0, 10.0F), -1,
                   "normal boundary native reject");
      std::cout << "PASS exact_boundary_rejects\n";
    }

    // UNIQUE PATH: every ancestor uses its own common pivot and sibling
    // boundary; the candidate reaches one leaf only.
    {
      std::vector<safe_c1_routing::Node> nodes(5);
      nodes[1] = safe_c1_routing::Node{101, 0.0F, false, false};
      nodes[2] = safe_c1_routing::Node{101, 10.0F, true, false};
      nodes[3] = safe_c1_routing::Node{202, 0.0F, true, false};
      nodes[4] = safe_c1_routing::Node{202, 5.0F, true, false};
      SearchNativeCertificate native(std::move(nodes), 2, 3);
      const int leaf = native.certify_leaf([](int pivot) {
        if (pivot == 101) return 2.0F;
        if (pivot == 202) return 2.0F;
        return std::numeric_limits<float>::quiet_NaN();
      });
      expect_equal(leaf, 3, "unique multi-level path");
      std::cout << "PASS unique_multi_level_path\n";
    }

    // NON-MONOTONIC SIBLING MINIMA: stale/corrupt ordering can make old own-max
    // routing accept one child while no search-native annulus proof exists.
    {
      std::vector<safe_c1_routing::Node> nodes(3);
      nodes[1] = safe_c1_routing::Node{101, 0.0F, true, false};
      nodes[2] = safe_c1_routing::Node{101, -1.0F, true, false};
      SearchNativeCertificate native(std::move(nodes), 2, 2);
      std::vector<OldMinMaxNode> old(3);
      old[1] = OldMinMaxNode{0.0F, 20.0F, false};
      old[2] = OldMinMaxNode{-1.0F, 5.0F, false};
      expect_equal(old_unique_minmax_child(old, 2, 0, 7.0F), 1,
                   "nonmonotonic old certificate accepts");
      expect_equal(native.only_strict_child(0, 7.0F), -1,
                   "nonmonotonic native rejects");
      std::cout << "PASS nonmonotonic_old_accepts_new_rejects\n";
    }

    // MISSING NEXT SIBLING: old min/max can accept child 1, but search_v2's
    // physical next-sibling boundary is unavailable. New certificate fails closed.
    {
      std::vector<safe_c1_routing::Node> nodes(3);
      nodes[1] = safe_c1_routing::Node{101, 0.0F, true, false};
      SearchNativeCertificate native(std::move(nodes), 2, 2);
      std::vector<OldMinMaxNode> old(3);
      old[1] = OldMinMaxNode{0.0F, 20.0F, false};
      expect_equal(old_unique_minmax_child(old, 2, 0, 7.0F), 1,
                   "missing sibling old certificate accepts");
      expect_equal(native.only_strict_child(0, 7.0F), -1,
                   "missing sibling native fails closed");
      std::cout << "PASS missing_sibling_old_accepts_new_rejects\n";
    }

    // FULL PHYSICAL TEN-SIBLING BLOCK: the production FrozenTreeSnapshot
    // requires TREE_ORDER==10 and rejects a missing physical sibling rather
    // than silently using a partial block.
    {
      std::vector<safe_c1_routing::Node> nodes(11);
      for (int slot = 0; slot < 10; ++slot) {
        nodes[static_cast<std::size_t>(slot + 1)] =
            safe_c1_routing::Node{303, static_cast<float>(slot * 10), true, false};
      }
      SearchNativeCertificate full(nodes, 10, 2);
      expect_equal(full.only_strict_child(0, 15.0F), 2,
                   "full ten-sibling native route");
      nodes[10].empty = true;
      SearchNativeCertificate missing(std::move(nodes), 10, 2);
      expect_equal(missing.only_strict_child(0, 15.0F), -1,
                   "full ten-sibling missing physical child rejects");
      std::cout << "PASS full_ten_sibling_contract_or_delta\n";
    }

    std::cout << "PASS safe_c1_search_native_routing_certificate CPU-only\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 2;
  }
}
