import unittest

import numpy as np

from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.utils import r, set_seed_r
from extreme_fit.distribution.gev.gev_params import GevParams


class TestStationaryGevFit(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)

    def test_stationary_gev_fit_with_ismev(self):
        params_estimated = fitted_stationary_gev(x_gev=np.array(r['x_gev']),
                                                 fit_method=MarginFitMethod.is_mev_gev_fit)
        ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8295}
        self.fit_estimator(params_estimated, ref)

    def test_stationary_gev_fit_with_mle(self):
        params_estimated = fitted_stationary_gev(x_gev=np.array(r['x_gev']),
                                                 fit_method=MarginFitMethod.extremes_fevd_mle)
        ref = {'loc': 0.02191974259369493, 'scale': 1.0347946062900268, 'shape': 0.829052520147379}
        self.fit_estimator(params_estimated, ref)

    def test_stationary_gev_fit_with_l_moments(self):
        params_estimated = fitted_stationary_gev(x_gev=np.array(r['x_gev']),
                                                 fit_method=MarginFitMethod.extremes_fevd_l_moments)
        ref = {'loc': 0.0813843045950251, 'scale': 1.1791830110181365, 'shape': 0.6610403806908737}
        self.fit_estimator(params_estimated, ref)

    def fit_estimator(self, params_estimated, ref):
        # Compare the MLE estimated parameters to the reference
        self.assertIsInstance(params_estimated, GevParams)
        params_estimated = params_estimated.to_dict()
        print(params_estimated)
        for key in ref.keys():
            self.assertAlmostEqual(ref[key], params_estimated[key], places=3)


if __name__ == '__main__':
    unittest.main()
