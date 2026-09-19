import unittest
from analyze import gmean, interval, matched, quantile


class EstimatorTests(unittest.TestCase):
    def test_geometric_not_arithmetic(self):
        self.assertAlmostEqual(gmean([.5, 2]), 1)

    def test_process_constant_interval(self):
        for value in interval([2] * 6):
            self.assertAlmostEqual(value, 2)

    def test_fixed_seed(self):
        self.assertEqual(interval([1, 2, 3, 4, 5, 6]), interval([1, 2, 3, 4, 5, 6]))

    def test_missing_cluster_rejected(self):
        with self.assertRaises(ValueError):
            interval([2] * 5)

    def test_unmatched_rejected(self):
        with self.assertRaises(ValueError):
            list(matched({1: {}}, {2: {}}))

    def test_counts_rejected(self):
        with self.assertRaises(ValueError):
            list(matched({1: {'counts': [1]}}, {1: {'counts': [2]}}))

    def test_quantile(self):
        self.assertEqual(quantile([1, 3], .5), 2)


if __name__ == '__main__':
    unittest.main()
