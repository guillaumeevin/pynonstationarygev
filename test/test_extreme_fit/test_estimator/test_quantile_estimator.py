import unittest

from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_margin import QuantileEstimatorFromMargin
from extreme_fit.estimator.quantile_estimator.quantile_estimator_from_regression import QuantileRegressionEstimator
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from test.test_utils import load_test_1D_and_2D_spatial_coordinates, load_test_spatiotemporal_coordinates, \
    load_smooth_margin_models, load_smooth_quantile_model_classes, load_test_temporal_coordinates


class TestQuantileEstimator(unittest.TestCase):
    DISPLAY = False

    def test_quantile_estimator_temporal(self):
        self.nb_points = 20
        self.nb_obs = 1
        self.coordinates = load_test_temporal_coordinates(nb_steps=self.nb_points)

        quantile = 0.5
        for coordinates in self.coordinates:
            constant_margin_model = StationaryTemporalModel(coordinates)
            dataset = MarginDataset.from_sampling(nb_obs=self.nb_obs,
                                                  margin_model=constant_margin_model,
                                                  coordinates=coordinates)
            # Load quantile estimators
            quantile_estimators = [
                QuantileEstimatorFromMargin(dataset, quantile, StationaryTemporalModel),
            ]
            for quantile_model_class in load_smooth_quantile_model_classes()[:]:
                quantile_estimator = QuantileRegressionEstimator(dataset, quantile, quantile_model_class)
                quantile_estimators.append(quantile_estimator)

            # Fit quantile estimators
            for quantile_estimator in quantile_estimators:
                quantile_estimator.fit()
                quantile_estimator.function_from_fit.visualize(show=self.DISPLAY)

        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
