import unittest

from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from test.test_utils import load_smooth_margin_models, load_test_1D_and_2D_coordinates


class TestSmoothMarginEstimator(unittest.TestCase):
    DISPLAY = False
    nb_points = 5

    def setUp(self):
        super().setUp()
        self.coordinates = load_test_1D_and_2D_coordinates(nb_points=self.nb_points)

    def test_dependency_estimators(self):
        for coordinates in self.coordinates:
            smooth_margin_models = load_smooth_margin_models(coordinates=coordinates)
            for margin_model in smooth_margin_models[1:]:
                dataset = MarginDataset.from_sampling(nb_obs=10,
                                                      margin_model=margin_model,
                                                      coordinates=coordinates)
                # Fit estimator
                estimator = SmoothMarginEstimator(dataset=dataset, margin_model=margin_model)
                estimator.fit()
                # Plot
                margin_model.margin_function_sample.visualize(show=self.DISPLAY)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
