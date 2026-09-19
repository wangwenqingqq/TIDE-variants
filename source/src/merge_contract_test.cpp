#include <iostream>
#include <stdexcept>
#include <vector>

#include "safe_c1_merge_contract.hpp"

int main() {
  using safe_c1_exporter::Candidate;
  using safe_c1_exporter::ExtraTierIds;
  using safe_c1_exporter::StableId;
  using safe_c1_exporter::merge_canonical_topk;

  try {
    ExtraTierIds extras;
    extras.sidecars_by_leaf.emplace(11, std::vector<StableId>{5, 6});
    extras.sidecars_by_leaf.emplace(12, std::vector<StableId>{7});
    extras.delta_ids = {8};
    extras.validate_storage(/*base_count=*/5, /*pool_count=*/9);

    // Only leaf 11 is exported by the frozen traversal: leaf 12 must not be
    // scanned as a sidecar candidate. Delta remains an exact global fallback.
    const auto selected = extras.select_for_visited_leaves(
        /*visited=*/{11, 11}, /*base_count=*/5, /*pool_count=*/9);
    if (selected.visited_sidecar_ids != std::vector<StableId>({5, 6}) ||
        selected.delta_ids != std::vector<StableId>({8})) {
      throw std::runtime_error("visited-leaf sidecar selection mismatch");
    }
    const std::vector<StableId> selected_ids = selected.combined();

    // Equal distances are ordered by stable ID regardless of tier.
    const std::vector<Candidate> ranked = merge_canonical_topk(
        /*base ids=*/{1, 3}, /*base distances=*/{1.0, 2.0},
        /*extra ids=*/selected_ids, /*extra distances=*/{0.5, 1.0, 3.0},
        /*k=*/2, /*base_count=*/5, /*pool_count=*/9);
    if (ranked.size() != 2 || ranked[0].stable_id != 5 || ranked[1].stable_id != 1 ||
        ranked[0].distance != 0.5 || ranked[1].distance != 1.0) {
      throw std::runtime_error("canonical merge/tie order mismatch");
    }

    bool rejected_duplicate = false;
    try {
      ExtraTierIds bad;
      bad.sidecars_by_leaf.emplace(11, std::vector<StableId>{5});
      bad.delta_ids = {5};
      bad.validate_storage(/*base_count=*/5, /*pool_count=*/9);
    } catch (const std::runtime_error&) {
      rejected_duplicate = true;
    }
    if (!rejected_duplicate) throw std::runtime_error("duplicate tier ID not rejected");

    std::cout << "PASS safe_c1_visited_leaf_merge_contract CPU-only\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 2;
  }
}
