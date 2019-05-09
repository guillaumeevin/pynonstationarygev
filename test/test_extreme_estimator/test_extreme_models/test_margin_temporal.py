import random
import unittest

import numpy as np
import pandas as pd

from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearNonStationaryLocationMarginModel, \
    LinearStationaryMarginModel
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import StationaryStationModel, \
    NonStationaryStationModel
from extreme_estimator.extreme_models.utils import r, set_seed_r, set_seed_for_test
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gev.ismev_gev_fit import IsmevGevFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from test.test_utils import load_test_spatiotemporal_coordinates, load_smooth_margin_models


class TestMarginTemporal(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_for_test(seed=42)
        self.nb_points = 2
        self.nb_steps = 50
        self.nb_obs = 1
        # Load some 2D spatial coordinates
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]
        smooth_margin_model = LinearNonStationaryLocationMarginModel(coordinates=self.coordinates,
                                                                      starting_point=2)
        self.dataset = MarginDataset.from_sampling(nb_obs=self.nb_obs,
                                                   margin_model=smooth_margin_model,
                                                   coordinates=self.coordinates)

    def test_margin_fit_stationary(self):
        # Create estimator
        margin_model = LinearStationaryMarginModel(self.coordinates)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        ref = {'loc': 1.3403346591679877, 'scale': 1.054867229157924, 'shape': 0.713700960727747}
        for year in range(1, 3):
            coordinate = np.array([0.0, 0.0, year])
            mle_params_estimated = estimator.margin_function_fitted.get_gev_params(coordinate).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

    def test_margin_fit_nonstationary(self):
        # Create estimator
        margin_model = LinearNonStationaryLocationMarginModel(self.coordinates)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        self.assertNotEqual(estimator.margin_function_fitted.mu1_temporal_trend, 0.0)
        # Checks that parameters returned are indeed different
        coordinate1 = np.array([0.0, 0.0, 1])
        mle_params_estimated_year1 = estimator.margin_function_fitted.get_gev_params(coordinate1).to_dict()
        coordinate3 = np.array([0.0, 0.0, 3])
        mle_params_estimated_year3 = estimator.margin_function_fitted.get_gev_params(coordinate3).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)

    def test_margin_fit_nonstationary_with_start_point(self):
        # Create estimator
        estimator = self.fit_non_stationary_estimator(starting_point=2)
        print(estimator.margin_function_fitted.mu1_temporal_trend)
        self.assertNotEqual(estimator.margin_function_fitted.mu1_temporal_trend, 0.0)
        # Checks starting point parameter are well passed
        self.assertEqual(2, estimator.margin_function_fitted.starting_point)
        # Checks that parameters returned are indeed different
        coordinate1 = np.array([0.0, 0.0, 1])
        mle_params_estimated_year1 = estimator.margin_function_fitted.get_gev_params(coordinate1).to_dict()
        coordinate2 = np.array([0.0, 0.0, 2])
        mle_params_estimated_year2 = estimator.margin_function_fitted.get_gev_params(coordinate2).to_dict()
        self.assertEqual(mle_params_estimated_year1, mle_params_estimated_year2)
        coordinate5 = np.array([0.0, 0.0, 5])
        mle_params_estimated_year5 = estimator.margin_function_fitted.get_gev_params(coordinate5).to_dict()
        self.assertNotEqual(mle_params_estimated_year5, mle_params_estimated_year2)

    def fit_non_stationary_estimator(self, starting_point):
        margin_model = LinearNonStationaryLocationMarginModel(self.coordinates, starting_point=starting_point)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        return estimator

    def test_two_different_starting_points(self):
        # Create two different estimators
        estimator1 = self.fit_non_stationary_estimator(starting_point=3)
        estimator2 = self.fit_non_stationary_estimator(starting_point=20)
        mu1_estimator1 = estimator1.margin_function_fitted.mu1_temporal_trend
        mu1_estimator2 = estimator2.margin_function_fitted.mu1_temporal_trend
        self.assertNotEqual(mu1_estimator1, mu1_estimator2)


if __name__ == '__main__':
    unittest.main()
