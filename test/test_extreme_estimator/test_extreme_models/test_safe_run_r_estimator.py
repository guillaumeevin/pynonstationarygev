import numpy as np
import unittest

from extreme_estimator.extreme_models.utils import safe_run_r_estimator, WarningMaximumAbsoluteValueTooHigh


def function(data=None, control=None):
    pass


class TestSafeRunREstimator(unittest.TestCase):

    def test_warning(self):
        threshold = 10
        value_above_threhsold = 2 * threshold
        datas = [np.array([value_above_threhsold]), np.ones([2, 2]) * value_above_threhsold]
        for data in datas:
            with self.assertWarns(WarningMaximumAbsoluteValueTooHigh):
                safe_run_r_estimator(function=function, data=data, threshold_max_abs_value=threshold)


if __name__ == '__main__':
    unittest.main()
