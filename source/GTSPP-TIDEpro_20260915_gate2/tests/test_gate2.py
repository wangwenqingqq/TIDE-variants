import unittest
from experiments.gate2.metrics import percentile, interval_for, passes, choose


class Gate2Protocol(unittest.TestCase):
    def test_nearest_rank_includes_tail(self):
        self.assertEqual(percentile(list(range(1, 101)), .99), 99)
        self.assertEqual(percentile([1, 1000], .99), 1000)

    def test_development_load_is_fixed_80_percent(self):
        self.assertAlmostEqual(interval_for(.8), 1.)
        self.assertGreaterEqual(interval_for(.807), .807/.8)
        self.assertEqual(interval_for(.01), .1)

    def case(self, policy, latency=2., coverage=True):
        return dict(policy=policy, p99_response_ms=latency, coverage_pass=coverage,
                    cpu_complete_match=True, completion_fraction=.99,
                    max_publication_lag_ms=1000., uploaded_bytes=100)

    def test_no_latency_win_can_hide_missing_coverage(self):
        self.assertEqual(choose([self.case("periodic2", .01, False), self.case("periodic3")], 4., "periodic"), "periodic3")

    def test_completion_and_update_deadline_are_required(self):
        c = self.case("tier2")
        self.assertTrue(passes(c, 2.))
        c["completion_fraction"] = .98999
        self.assertFalse(passes(c, 2.))
        c["completion_fraction"] = 1.
        c["max_publication_lag_ms"] = 1000.001
        self.assertFalse(passes(c, 2.))

    def test_deterministic_parameter_tie_break(self):
        self.assertEqual(choose([self.case("tier4"), self.case("tier2")], 2., "tier"), "tier2")


if __name__ == "__main__":
    unittest.main()
