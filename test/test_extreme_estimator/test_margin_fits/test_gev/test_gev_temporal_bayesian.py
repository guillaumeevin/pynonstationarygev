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
        ref = {'loc': 0.082261, 'scale': 1.183703, 'shape': 0.882750}
        for year in range(1, 3):
            mle_params_estimated = estimator_fitted.margin_function_from_fit.get_gev_params(np.array([year])).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

    # def test_gev_temporal_margin_fit_non_stationary(self):
    #     # Create estimator
    #     estimator_fitted = fitted_linear_margin_estimator(NonStationaryLocationStationModel, self.coordinates, self.dataset,
    #                                                       starting_year=0,
    #                                                       fit_method=AbstractTemporalLinearMarginModel.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR)
    #     result_from_model_fit = estimator_fitted.result_from_model_fit  # type: ResultFromExtremes
    #     print(result_from_model_fit.posterior_samples)

    # print(estimator.result_from_model_fit.r)
    # ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8295}
    # for year in range(1, 3):
    #     mle_params_estimated = estimator.margin_function_from_fit.get_gev_params(np.array([year])).to_dict()
    #     for key in ref.keys():
    #         self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)


if __name__ == '__main__':
    unittest.main()
