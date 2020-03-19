import numpy as np
import unittest

from extreme_fit.model.utils import safe_run_r_estimator, WarningMaximumAbsoluteValueTooHigh, WarningTooMuchZeroValues


def empty_function(data=None, control=None):
    pass


class TestSafeRunREstimator(unittest.TestCase):

    def test_warning_maximum_value(self):
        ratio = 10
        data = np.array([ratio+1, 1])
        with self.assertWarns(WarningMaximumAbsoluteValueTooHigh):
            safe_run_r_estimator(function=empty_function, data=data, max_ratio_between_two_extremes_values=ratio)

    def test_warning_too_much_zero(self):
        n = 5
        data = np.concatenate([np.zeros(n), np.ones(n)])
        with self.assertWarns(WarningTooMuchZeroValues):
            safe_run_r_estimator(function=empty_function, data=data)


if __name__ == '__main__':
    unittest.main()
