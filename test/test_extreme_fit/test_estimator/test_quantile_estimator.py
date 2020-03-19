import unittest

from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import QuantileEstimatorFromMargin, \
    QuantileRegressionEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from test.test_utils import load_test_1D_and_2D_spatial_coordinates, load_test_spatiotemporal_coordinates, \
    load_smooth_margin_models, load_smooth_quantile_model_classes


class TestQuantileEstimator(unittest.TestCase):
    DISPLAY = False

    def test_smooth_margin_estimator_spatial(self):
        self.nb_points = 20
        self.nb_obs = 1
        self.coordinates = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points)[:1]

    def test_smooth_margin_estimator_spatio_temporal(self):
        self.nb_points = 2
        self.nb_steps = 2
        self.nb_obs = 1
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)

    def tearDown(self) -> None:
        quantile = 0.5
        for coordinates in self.coordinates:
            constant_margin_model = load_smooth_margin_models(coordinates=coordinates)[0]
            dataset = MarginDataset.from_sampling(nb_obs=self.nb_obs,
                                                  margin_model=constant_margin_model,
                                                  coordinates=coordinates)
            # Load quantile estimators
            quantile_estimators = [
                QuantileEstimatorFromMargin(dataset, constant_margin_model, quantile),
            ]
            for quantile_model_class in load_smooth_quantile_model_classes()[:1]:
                quantile_estimator = QuantileRegressionEstimator(dataset, quantile, quantile_model_class)
                quantile_estimators.append(quantile_estimator)

            # Fit quantile estimators
            for quantile_estimator in quantile_estimators:
                quantile_estimator.fit()
                quantile_estimator.function_from_fit.visualize(show=self.DISPLAY)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
