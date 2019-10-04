import unittest

import numpy as np

from extreme_estimator.extreme_models.utils import r, set_seed_r
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gev.ismev_gev_fit import IsmevGevFit


class TestGevMleFit(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)

    def test_gevmle_fit(self):
        estimator = GevMleFit(x_gev=np.array(r['x_gev']))
        mle_params_ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8290}
        self.fit_estimator(estimator, ref=mle_params_ref)

    def test_ismev_gev_fit(self):
        estimator = IsmevGevFit(x_gev=np.array(r['x_gev']))
        ismev_ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8295}
        self.fit_estimator(estimator, ismev_ref)

    # def test_s

    def fit_estimator(self, estimator, ref):
        # Compare the MLE estimated parameters to the reference
        mle_params_estimated = estimator.gev_params
        self.assertIsInstance(mle_params_estimated, GevParams)
        mle_params_estimated = mle_params_estimated.to_dict()
        for key in ref.keys():
            self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)


if __name__ == '__main__':
    unittest.main()
