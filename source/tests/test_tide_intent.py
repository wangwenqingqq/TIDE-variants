import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from page2tile.tide_intent import eligible_popcount_range, summarize_windows  # noqa: E402


class TideIntentTest(unittest.TestCase):
    def test_exact_population_count_range(self) -> None:
        self.assertEqual(eligible_popcount_range(54, 7, 10), (38, 77))
        self.assertEqual(eligible_popcount_range(54, 4, 5), (44, 67))
        self.assertEqual(eligible_popcount_range(256, 4, 5), (205, 256))

    def test_pair_weighted_fanin_and_grouping(self) -> None:
        bins = [0] * 257
        bins[1] = 10
        bins[2] = 20
        result = summarize_windows([1, 1], bins, 1, 1, 2)
        aggregate = result["aggregate"]
        self.assertEqual(aggregate["candidate_query_pairs"], 20)
        self.assertEqual(aggregate["logical_query_major_fingerprint_bytes"], 640)
        self.assertEqual(aggregate["logical_grouped_once_fingerprint_bytes"], 320)
        self.assertEqual(aggregate["logical_grouping_ratio"], 2.0)
        self.assertEqual(aggregate["pair_fraction_by_min_fanin"]["2"], 1.0)
        self.assertEqual(aggregate["pair_fraction_by_min_fanin"]["4"], 0.0)

    def test_adjacent_stability_is_well_formed(self) -> None:
        bins = [0] * 257
        bins[10] = 4
        result = summarize_windows([10, 10], bins, 1, 1, 1)
        stability = result["aggregate"]["adjacent_window_active_bin_jaccard"]
        self.assertEqual(stability["n"], 1)
        self.assertEqual(stability["mean"], 1.0)


if __name__ == "__main__":
    unittest.main()
