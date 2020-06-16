import unittest

import numpy as np
import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model import NonStationaryQuadraticLocationModel
from extreme_trend.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalQuadraticExtremesMle(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 1
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
        self.fit_method = MarginFitMethod.extremes_fevd_mle

    def test_gev_temporal_margin_fit_non_stationary_quadratic_location(self):
        # Create estimator
        estimator = fitted_linear_margin_estimator(NonStationaryQuadraticLocationModel,
                                                   self.coordinates, self.dataset,
                                                   starting_year=0,
                                                   fit_method=self.fit_method)
        # Checks that parameters returned are indeed different
        mle_params_estimated_year1 = estimator.function_from_fit.get_params(np.array([1])).to_dict()
        mle_params_estimated_year3 = estimator.function_from_fit.get_params(np.array([21])).to_dict()
        mle_params_estimated_year5 = estimator.function_from_fit.get_params(np.array([41])).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)
        self.assertNotEqual(mle_params_estimated_year3, mle_params_estimated_year5)
        # Assert the relationship for the location is indeed quadratic
        location_gev_params = GevParams.LOC
        diff1 = mle_params_estimated_year1[location_gev_params] - mle_params_estimated_year3[location_gev_params]
        diff2 = mle_params_estimated_year3[location_gev_params] - mle_params_estimated_year5[location_gev_params]
        self.assertNotAlmostEqual(diff1, diff2)
        # Assert that indicators are correctly computed
        self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh())
        # self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic())
        # self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic())


if __name__ == '__main__':
    unittest.main()
