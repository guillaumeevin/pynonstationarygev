import unittest

import numpy as np

from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.param_function import LinearParamFunction


class TestParamFunction(unittest.TestCase):

    # def test_out_of_bounds(self):
    #     param_function = LinearParamFunction(dims=[0], coordinates=np.array([[0]]), linear_coef=LinearCoef())
    #     with self.assertRaises(AssertionError):
    #         param_function.get_param_value(np.array([1.0]))

    def test_linear_param_function(self):
        linear_coef = LinearCoef(idx_to_coef={0: 1})
        param_function = LinearParamFunction(dims=[0], coordinates=np.array([[-1, 0, 1]]).transpose(),
                                             linear_coef=linear_coef)
        self.assertEqual(0.0, param_function.get_param_value(np.array([0.0])))
        self.assertEqual(1.0, param_function.get_param_value(np.array([1.0])))

    def test_affine_param_function(self):
        linear_coef = LinearCoef(idx_to_coef={-1: 1, 0: 1})
        param_function = LinearParamFunction(dims=[0], coordinates=np.array([[-1, 0, 1]]).transpose(),
                                             linear_coef=linear_coef)
        self.assertEqual(1.0, param_function.get_param_value(np.array([0.0])))
        self.assertEqual(2.0, param_function.get_param_value(np.array([1.0])))


if __name__ == '__main__':
    unittest.main()
