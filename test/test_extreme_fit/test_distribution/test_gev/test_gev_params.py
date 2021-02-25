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

    def test_wrapper(self):
        gev_params = GevParams(loc=1.0, shape=1.0, scale=-1.0)
        self.assertTrue(np.isnan(gev_params.quantile(p=0.5)))
        self.assertTrue(np.isnan(gev_params.sample(n=10)))
        self.assertTrue(np.isnan(gev_params.param_values))
        self.assertTrue(np.isnan(gev_params.density(x=1.5)))

    def test_time_derivative_return_level(self):
        p = 0.99
        for mu1 in [-1, 0, 1]:
            for sigma1 in [0, 1, 10]:
                for shape in [-1, 0, 1]:
                    params = GevParams(loc=mu1, scale=sigma1, shape=shape, accept_zero_scale_parameter=True)
                    quantile = params.quantile(p)
                    time_derivative = params.time_derivative_of_return_level(p, mu1, sigma1)
                    self.assertEqual(quantile, time_derivative)

    def test_gumbel_standardization(self):
        standard_gumbel = GevParams(0, 1, 0)
        x = standard_gumbel.sample(10)
        for shift in [-1, 0, 1]:
            for scale in [1, 10]:
                x_shifted_and_scaled = (x * scale) + shift
                gumbel = GevParams(shift, scale, 0)
                x_standardized = gumbel.gumbel_standardization(x_shifted_and_scaled)
                np.testing.assert_almost_equal(x, x_standardized)
                x_inverse_standardization = gumbel.gumbel_inverse_standardization(x_standardized)
                np.testing.assert_almost_equal(x_shifted_and_scaled, x_inverse_standardization)

    def test_negative_scale(self):
        gev_params = GevParams(loc=1.0, shape=1.0, scale=-1.0)
        for p in [0.1, 0.5, 0.9]:
            q = gev_params.quantile(p)
            self.assertTrue(gev_params.has_undefined_parameters)
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

    def test_return_level_plot(self):
        gev_params = GevParams(loc=0.0, shape=0.0, scale=1.0)
        gev_params.return_level_plot_against_return_period(show=False)


if __name__ == '__main__':
    unittest.main()
