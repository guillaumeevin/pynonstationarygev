import unittest

import numpy as np

from extreme_estimator.margin_fits.gev.gev_params import GevParams


class TestGevParams(unittest.TestCase):

    def test_quantile(self):
        # For GEV(1,1,1), the repartition function is exp(-y^-1) the formula for the quantile p is -1/log(p)
        gev_params = GevParams(loc=1.0, shape=1.0, scale=1.0)
        quantile_dict = gev_params.quantile_name_to_value
        for quantile_name, p in gev_params.quantile_name_to_p.items():
            self.assertAlmostEqual(- 1 / np.log(p), quantile_dict[quantile_name])


if __name__ == '__main__':
    unittest.main()
