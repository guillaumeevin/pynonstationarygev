import unittest

import numpy as np
import pandas as pd

from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryStationModel, \
    NonStationaryLocationStationModel
from extreme_fit.model.result_from_model_fit.result_from_extremes import ResultFromExtremes
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from test.test_utils import load_non_stationary_temporal_margin_models


class TestGevTemporalBayesian(unittest.TestCase):

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
        self.fit_method = AbstractTemporalLinearMarginModel.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR

    def test_gev_temporal_margin_fit_stationary(self):
        # Create estimator
        estimator_fitted = fitted_linear_margin_estimator(StationaryStationModel, self.coordinates, self.dataset,
                                                          starting_year=0,
                                                          fit_method=AbstractTemporalLinearMarginModel.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR)
        ref = {'loc': 0.30440183718071845, 'scale': 1.301639258861012, 'shape': 0.303652401064773}
        for year in range(1, 3):
            mle_params_estimated = estimator_fitted.margin_function_from_fit.get_gev_params(np.array([year])).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

    def test_gev_temporal_margin_fit_non_stationary(self):
        # Create estimator
        estimator = fitted_linear_margin_estimator(NonStationaryLocationStationModel, self.coordinates, self.dataset,
                                                   starting_year=0,
                                                   fit_method=AbstractTemporalLinearMarginModel.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR)
        mu1_values = estimator.result_from_model_fit.df_posterior_samples.iloc[:, 1]
        self.assertTrue((mu1_values != 0).any())
        # Checks that parameters returned are indeed different
        mle_params_estimated_year1 = estimator.margin_function_from_fit.get_gev_params(np.array([1])).to_dict()
        mle_params_estimated_year3 = estimator.margin_function_from_fit.get_gev_params(np.array([3])).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)

    #
    # def test_gev_temporal_margin_fit_nonstationary_with_start_point(self):
    #     # Create estimator
    #     estimator = self.fit_non_stationary_estimator(starting_point=3)
    #     self.assertNotEqual(estimator.margin_function_from_fit.mu1_temporal_trend, 0.0)
    #     # Checks starting point parameter are well passed
    #     self.assertEqual(3, estimator.margin_function_from_fit.starting_point)
    #     # Checks that parameters returned are indeed different
    #     mle_params_estimated_year1 = estimator.margin_function_from_fit.get_gev_params(np.array([1])).to_dict()
    #     mle_params_estimated_year3 = estimator.margin_function_from_fit.get_gev_params(np.array([3])).to_dict()
    #     self.assertEqual(mle_params_estimated_year1, mle_params_estimated_year3)
    #     mle_params_estimated_year5 = estimator.margin_function_from_fit.get_gev_params(np.array([5])).to_dict()
    #     self.assertNotEqual(mle_params_estimated_year5, mle_params_estimated_year3)
    #
    # def fit_non_stationary_estimator(self, starting_point):
    #     margin_model = NonStationaryLocationStationModel(self.coordinates,
    #                                                      starting_point=starting_point + self.start_year)
    #     estimator = LinearMarginEstimator(self.dataset, margin_model)
    #     estimator.fit()
    #     return estimator
    #
    # def test_two_different_starting_points(self):
    #     # Create two different estimators
    #     estimator1 = self.fit_non_stationary_estimator(starting_point=3)
    #     estimator2 = self.fit_non_stationary_estimator(starting_point=28)
    #     mu1_estimator1 = estimator1.margin_function_from_fit.mu1_temporal_trend
    #     mu1_estimator2 = estimator2.margin_function_from_fit.mu1_temporal_trend
    #     self.assertNotEqual(mu1_estimator1, mu1_estimator2)


if __name__ == '__main__':
    unittest.main()
