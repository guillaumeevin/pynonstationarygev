import unittest

from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from test.test_utils import load_smooth_margin_models, load_test_1D_and_2D_spatial_coordinates, \
    load_test_spatiotemporal_coordinates


class TestSmoothMarginEstimator(unittest.TestCase):
    DISPLAY = False

    def test_smooth_margin_estimator_spatial(self):
        self.nb_points = 2
        self.nb_obs = 2
        self.coordinates = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points)

    def test_smooth_margin_estimator_spatio_temporal(self):
        self.nb_points = 2
        self.nb_steps = 2
        self.nb_obs = 1
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)

    def tearDown(self) -> None:
        for coordinates in self.coordinates:
            smooth_margin_models = load_smooth_margin_models(coordinates=coordinates)
            for margin_model in smooth_margin_models[1:]:
                dataset = MarginDataset.from_sampling(nb_obs=self.nb_obs,
                                                      margin_model=margin_model,
                                                      coordinates=coordinates)
                # Fit estimator
                estimator = LinearMarginEstimator(dataset=dataset, margin_model=margin_model)
                estimator.fit()
                print(estimator.result_from_model_fit.name_to_value.keys())
                # Plot
                if self.DISPLAY:
                    margin_model.margin_function.visualize_function(show=True)
                    estimator.function_from_fit.visualize_function(show=True)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
