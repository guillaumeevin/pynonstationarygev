import unittest

import numpy as np

from extreme_fit.estimator.full_estimator.abstract_full_estimator import \
    FullEstimatorInASingleStepWithSmoothMargin
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearNonStationaryLocationMarginModel, \
    LinearStationaryMarginModel
from extreme_fit.model.utils import set_seed_for_test
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from test.test_utils import load_test_spatiotemporal_coordinates, load_test_max_stable_models


class TestMaxStableTemporal(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_for_test(seed=42)
        self.nb_points = 2
        self.nb_steps = 50
        self.nb_obs = 1
        # Load some 2D spatial coordinates
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]
        self.smooth_margin_model = LinearNonStationaryLocationMarginModel(coordinates=self.coordinates,
                                                                          starting_point=2)
        self.max_stable_model = load_test_max_stable_models()[0]
        self.dataset = FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs,
                                                                 margin_model=self.smooth_margin_model,
                                                                 coordinates=self.coordinates,
                                                                 max_stable_model=self.max_stable_model)

    def test_margin_fit_stationary(self):
        # Create estimator
        margin_model = LinearStationaryMarginModel(self.coordinates)
        estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model,
                                                               self.max_stable_model)
        estimator.fit()
        ref = {'loc': 1.2091156634312243, 'scale': 1.1210085591373455, 'shape': 0.9831957705294134}
        for year in range(1, 3):
            coordinate = np.array([0.0, 0.0, year])
            mle_params_estimated = estimator.margin_function_from_fit.get_params(coordinate).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

    def test_margin_fit_nonstationary(self):
        # Create estimator
        margin_model = LinearNonStationaryLocationMarginModel(self.coordinates)
        estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model,
                                                               self.max_stable_model)
        estimator.fit()
        self.assertNotEqual(estimator.margin_function_from_fit.mu1_temporal_trend, 0.0)
        # Checks that parameters returned are indeed different
        coordinate1 = np.array([0.0, 0.0, 1])
        mle_params_estimated_year1 = estimator.margin_function_from_fit.get_params(coordinate1).to_dict()
        coordinate3 = np.array([0.0, 0.0, 3])
        mle_params_estimated_year3 = estimator.margin_function_from_fit.get_params(coordinate3).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)

    def test_margin_fit_nonstationary_with_start_point(self):
        # Create estimator
        estimator = self.fit_non_stationary_estimator(starting_point=2)
        # By default, estimator find the good margin
        self.assertNotEqual(estimator.margin_function_from_fit.mu1_temporal_trend, 0.0)
        self.assertAlmostEqual(estimator.margin_function_from_fit.mu1_temporal_trend,
                               self.smooth_margin_model.margin_function.mu1_temporal_trend,
                               places=2)
        # Checks starting point parameter are well passed
        self.assertEqual(2, estimator.margin_function_from_fit.starting_point)
        # Checks that parameters returned are indeed different
        coordinate1 = np.array([0.0, 0.0, 1])
        mle_params_estimated_year1 = estimator.margin_function_from_fit.get_params(coordinate1).to_dict()
        coordinate2 = np.array([0.0, 0.0, 2])
        mle_params_estimated_year2 = estimator.margin_function_from_fit.get_params(coordinate2).to_dict()
        self.assertEqual(mle_params_estimated_year1, mle_params_estimated_year2)
        coordinate5 = np.array([0.0, 0.0, 5])
        mle_params_estimated_year5 = estimator.margin_function_from_fit.get_params(coordinate5).to_dict()
        self.assertNotEqual(mle_params_estimated_year5, mle_params_estimated_year2)

    def fit_non_stationary_estimator(self, starting_point):
        margin_model = LinearNonStationaryLocationMarginModel(self.coordinates, starting_point=starting_point)
        estimator = FullEstimatorInASingleStepWithSmoothMargin(self.dataset, margin_model,
                                                               self.max_stable_model)
        estimator.fit()
        return estimator

    def test_two_different_starting_points(self):
        # Create two different estimators
        estimator1 = self.fit_non_stationary_estimator(starting_point=3)
        estimator2 = self.fit_non_stationary_estimator(starting_point=20)
        mu1_estimator1 = estimator1.margin_function_from_fit.mu1_temporal_trend
        mu1_estimator2 = estimator2.margin_function_from_fit.mu1_temporal_trend
        self.assertNotEqual(mu1_estimator1, mu1_estimator2)


if __name__ == '__main__':
    unittest.main()
