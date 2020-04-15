import unittest

import numpy as np
import pandas as pd

from extreme_trend.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import GumbelTemporalModel
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalExtremesGumbel(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 0
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)
        # Compute the stationary temporal margin with isMev
        self.start_year = 0
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: range(self.start_year, self.start_year + 50)})
        self.coordinates = AbstractTemporalCoordinates.from_df(df)
        df2 = pd.DataFrame(data=np.array(r['x_gev']), index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)

    def test_gev_temporal_margin_fit(self):
        # Create estimator
        estimator = fitted_linear_margin_estimator(GumbelTemporalModel,
                                                   self.coordinates, self.dataset,
                                                   starting_year=0,
                                                   fit_method=MarginFitMethod.extremes_fevd_mle)
        ref = {'loc': -0.0862185692806497, 'scale': 1.0818465357627252, 'shape': 0}
        for year in range(1, 3):
            mle_params_estimated = estimator.function_from_fit.get_gev_params(np.array([year])).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

if __name__ == '__main__':
    unittest.main()
