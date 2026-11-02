import unittest

from report import regression_status


class RegressionStatusTests(unittest.TestCase):
    def test_missing_baseline_is_unknown(self):
        self.assertIsNone(regression_status(1.0, None))

    def test_latency_over_ten_percent_is_regression(self):
        self.assertTrue(regression_status(1.11, 1.0))

    def test_latency_at_ten_percent_is_not_regression(self):
        self.assertFalse(regression_status(1.1, 1.0))


if __name__ == "__main__":
    unittest.main()