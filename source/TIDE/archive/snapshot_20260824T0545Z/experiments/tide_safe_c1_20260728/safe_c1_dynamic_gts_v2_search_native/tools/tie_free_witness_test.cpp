#include <cstdint>
#include <exception>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "safe_c1_tie_free_witness.hpp"

namespace {

void expect_abort(const std::function<void()>& fn, const std::string& token,
                  const std::string& label) {
  try {
    fn();
  } catch (const std::runtime_error& error) {
    const std::string message(error.what());
    if (message.find(token) != std::string::npos) return;
    throw std::runtime_error(label + ": wrong error: " + message);
  }
  throw std::runtime_error(label + ": expected abort was not raised");
}

std::vector<std::int16_t> float_collision_pool() {
  // id=2 is zero. IDs 0/1 have exact squared keys differing by one, but the
  // legacy GTS float key collapses them. Their collision can be either outside
  // or inside the inspected static top-(k+1) prefix depending on k.
  constexpr int dimension = 128;
  std::vector<std::int16_t> pool(static_cast<std::size_t>(3 * dimension), 0);
  for (int d = 0; d < dimension - 1; ++d) {
    pool[static_cast<std::size_t>(d)] = 32767;
    pool[static_cast<std::size_t>(dimension + d)] = 32767;
  }
  pool[static_cast<std::size_t>(dimension + dimension - 1)] = 1;
  return pool;
}

}  // namespace

int main() {
  try {
    const std::vector<std::int16_t> zero_query{0};

    {
      // k=2 inspects keys 1,4,9. The far keys 100,100 may tie because they
      // cannot alter the static top-k or k+1 boundary.
      const std::vector<std::int16_t> base_with_far_tie{1, 2, 3, 10, -10};
      safe_c1_tie::TieFreeWitnessChecker checker(
          base_with_far_tie, zero_query, 1, 5, 2);
      checker.require_static_base_top_k_plus_one_tie_free(nullptr);
      std::cout << "PASS tie_free_witness_allows_static_exact_tie_after_top_k_plus_one\n";
    }

    {
      // k=2 inspects keys 1,1,25, so an internal top-k tie is rejected.
      const std::vector<std::int16_t> prefix_exact_tie{1, -1, 5, 10};
      safe_c1_tie::TieFreeWitnessChecker checker(prefix_exact_tie, zero_query, 1, 4, 2);
      expect_abort([&] { checker.require_static_base_top_k_plus_one_tie_free(nullptr); },
                   "static_base_top_k_plus_one_exact_squared_l2_tie",
                   "static exact prefix tie");
      std::cout << "PASS tie_free_witness_aborts_static_exact_tie_in_top_k_plus_one\n";
    }

    {
      constexpr int dimension = 128;
      const std::vector<std::int16_t> pool = float_collision_pool();
      const std::vector<std::int16_t> query(static_cast<std::size_t>(dimension), 0);
      // k=1 inspects zero and the first huge vector. The modeled-float
      // collision is at ranks 1/2, strictly after the inspected prefix.
      safe_c1_tie::TieFreeWitnessChecker checker(pool, query, dimension, 3, 1);
      checker.require_static_base_top_k_plus_one_tie_free(nullptr);
      std::cout << "PASS tie_free_witness_allows_static_modeled_float_tie_after_top_k_plus_one\n";
    }

    {
      constexpr int dimension = 128;
      const std::vector<std::int16_t> pool = float_collision_pool();
      const std::vector<std::int16_t> query(static_cast<std::size_t>(dimension), 0);
      // k=2 inspects all three base objects; the modeled GTS float collision is
      // now inside top-(k+1), even though exact squared keys are distinct.
      safe_c1_tie::TieFreeWitnessChecker checker(pool, query, dimension, 3, 2);
      expect_abort([&] { checker.require_static_base_top_k_plus_one_tie_free(nullptr); },
                   "static_base_top_k_plus_one_modeled_gts_float_key_tie",
                   "static modeled-float prefix tie");
      std::cout << "PASS tie_free_witness_aborts_static_modeled_float_tie_in_top_k_plus_one\n";
    }

    {
      // Dynamic IDs 3 and 4 tie internally at key 4. With k=3, the dynamic
      // k/k+1 boundary is 4/16, so the internal tie remains permitted.
      const std::vector<std::int16_t> pool{1, 4, 5, 2, 2};
      const std::vector<std::int16_t> queries{0};
      safe_c1_tie::TieFreeWitnessChecker checker(pool, queries, 1, 3, 3);
      safe_c1_tie::WitnessAudit audit;
      checker.require_static_base_top_k_plus_one_tie_free(&audit);
      const std::vector<std::uint8_t> active{1, 1, 1, 1, 1};
      checker.require_dynamic_k_boundary_tie_free(active, 0, &audit);
      if (audit.static_base_queries_checked != 1 ||
          audit.dynamic_k_boundary_queries_checked != 1) {
        throw std::runtime_error("witness audit counter mismatch");
      }
      std::cout << "PASS tie_free_witness_allows_dynamic_internal_tie_away_from_boundary\n";
    }

    {
      const std::vector<std::int16_t> pool{1, 4, 5, 2, 2};
      const std::vector<std::int16_t> queries{0};
      safe_c1_tie::TieFreeWitnessChecker checker(pool, queries, 1, 3, 2);
      const std::vector<std::uint8_t> active{1, 1, 1, 1, 1};
      // Sorted active keys are 1,4,4,16,25, so k=2 has a membership-ambiguous tie.
      expect_abort([&] { checker.require_dynamic_k_boundary_tie_free(active, 0, nullptr); },
                   "dynamic_k_boundary_distance_tie", "dynamic boundary tie");
      std::cout << "PASS tie_free_witness_aborts_dynamic_k_boundary_tie\n";
    }

    std::cout << "PASS safe_c1_tie_free_witness CPU-only\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << '\n';
    return 2;
  }
}
