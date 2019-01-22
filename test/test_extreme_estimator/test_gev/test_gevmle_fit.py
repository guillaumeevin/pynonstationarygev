import unittest

import numpy as np

from extreme_estimator.extreme_models.utils import r, set_seed_r
from extreme_estimator.gev.gevmle_fit import GevMleFit


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
        # Get the MLE estimator
        estimator = GevMleFit(x_gev=np.array(r['x_gev']))
        self.fit_estimator(estimator)

    def fit_estimator(self, estimator):
        # Compare the MLE estimated parameters to the reference
        mle_params_estimated = estimator.mle_params
        mle_params_ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8290}
        for key in mle_params_ref.keys():
            self.assertAlmostEqual(mle_params_ref[key], mle_params_estimated[key], places=3)


if __name__ == '__main__':
    unittest.main()
