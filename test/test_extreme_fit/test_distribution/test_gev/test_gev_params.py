import unittest

from mpmath import euler

import numpy as np
from scipy.special.cython_special import gamma

from extreme_fit.distribution.gev.gev_params import GevParams


class TestGevParams(unittest.TestCase):

    def test_quantile(self):
        # For GEV(1,1,1), the repartition function is exp(-y^-1) the formula for the quantile p is -1/log(p)
        gev_params = GevParams(loc=1.0, shape=1.0, scale=1.0)
        quantile_dict = gev_params.quantile_name_to_value
        for quantile_name, p in gev_params.quantile_name_to_p.items():
            self.assertAlmostEqual(- 1 / np.log(p), quantile_dict[quantile_name])

    def test_negative_scale(self):
        gev_params = GevParams(loc=1.0, shape=1.0, scale=-1.0)
        for p in [0.1, 0.5, 0.9]:
            q = gev_params.quantile(p)
            self.assertTrue(np.isnan(q))

    def test_has_undefined_parameter(self):
        gev_params = GevParams(loc=1.0, shape=1.0, scale=-1.0)
        self.assertTrue(gev_params.has_undefined_parameters)
        for k, v in gev_params.indicator_name_to_value.items():
            self.assertTrue(np.isnan(v), msg="{} is not equal to np.nan".format(k))

    def test_limit_cases(self):
        gev_params = GevParams(loc=1.0, shape=1.0, scale=1.0)
        self.assertEqual(gev_params.mean, np.inf)
        gev_params = GevParams(loc=1.0, shape=0.5, scale=1.0)
        self.assertEqual(gev_params.variance, np.inf)
        self.assertEqual(gev_params.std, np.inf)

    def test_mean(self):
        mu = 1.0
        sigma = 1.0
        gev_params = GevParams(loc=mu, shape=0.0, scale=sigma)
        self.assertEqual(gev_params.mean, mu + sigma * euler)
        chi = 0.5
        gev_params = GevParams(loc=mu, shape=chi, scale=sigma)
        self.assertEqual(gev_params.mean, mu + sigma * (gamma(1 - 0.5) - 1) / chi)

    def test_variance(self):
        mu = 1.0
        sigma = 1.0
        gev_params = GevParams(loc=mu, shape=0.0, scale=sigma)
        self.assertEqual(gev_params.variance, ((sigma * np.math.pi) ** 2) / 6)
        chi = 0.25
        gev_params = GevParams(loc=mu, shape=chi, scale=sigma)
        self.assertEqual(gev_params.variance, ((sigma / chi) ** 2) * (gamma(1 - 2 * chi) - (gamma(1 - chi) ** 2)))


if __name__ == '__main__':
    unittest.main()