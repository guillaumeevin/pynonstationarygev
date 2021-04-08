import unittest

import numpy as np
import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import \
    NonStationaryQuadraticLocationModel, \
    NonStationaryQuadraticScaleModel, NonStationaryQuadraticLocationGumbelModel, NonStationaryQuadraticScaleGumbelModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationModel, NonStationaryTwoLinearScaleModel
from extreme_trend.trend_test.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalSpline(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 51
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)
        # Compute the stationary temporal margin with isMev
        self.start_year = -25
        nb_years = 51
        self.last_year = self.start_year + nb_years - 1
        years = np.array(range(self.start_year, self.start_year + nb_years))
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: years})
        self.coordinates = AbstractTemporalCoordinates.from_df(df)
        df2 = pd.DataFrame(data=np.array(r['x_gev']), index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        self.fit_method = MarginFitMethod.evgam

    def function_test_gev_temporal_margin_fit_non_stationary_spline(self, model_class, param_to_test):
        # Create estimator
        estimator = fitted_linear_margin_estimator(model_class,
                                                   self.coordinates, self.dataset,
                                                   starting_year=None,
                                                   fit_method=self.fit_method)
        # Checks that parameters returned are indeed different
        mle_params_estimated_year1 = estimator.function_from_fit.get_params(np.array([0])).to_dict()
        mle_params_estimated_year3 = estimator.function_from_fit.get_params(np.array([21])).to_dict()
        mle_params_estimated_year5 = estimator.function_from_fit.get_params(np.array([self.last_year])).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)
        self.assertNotEqual(mle_params_estimated_year3, mle_params_estimated_year5)
        # # Assert the relationship for the location is different between the beginning and the end
        diff1 = mle_params_estimated_year1[param_to_test] - mle_params_estimated_year3[param_to_test]
        diff2 = mle_params_estimated_year3[param_to_test] - mle_params_estimated_year5[param_to_test]
        self.assertNotAlmostEqual(diff1, diff2)

        for idx, nb_year in enumerate(range(5)):
            year = self.start_year + nb_year
            gev_params_from_result = estimator.result_from_model_fit.get_gev_params_from_result(idx).to_dict()
            my_gev_params = estimator.function_from_fit.get_params(np.array([year])).to_dict()
            for param_name in GevParams.PARAM_NAMES:
                self.assertAlmostEqual(gev_params_from_result[param_name], my_gev_params[param_name],
                                       msg='for the {} parameter at year={}'.format(param_name, year),
                                       places=2)
        # Assert that indicators are correctly computed
        self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh())
        # self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic())
        # self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic())

    def test_gev_temporal_margin_fit_spline_two_linear_location(self):
        self.function_test_gev_temporal_margin_fit_non_stationary_spline(NonStationaryTwoLinearLocationModel,
                                                                         param_to_test=GevParams.LOC)

    def test_gev_temporal_margin_fit_spline_two_linear_scale(self):
        self.function_test_gev_temporal_margin_fit_non_stationary_spline(NonStationaryTwoLinearScaleModel,
                                                                         param_to_test=GevParams.SCALE)


if __name__ == '__main__':
    unittest.main()
