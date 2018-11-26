import unittest

from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.return_level_plot.spatial_2D_plot import Spatial2DPlot
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
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
            for margin_model in smooth_margin_models:
                dataset = MarginDataset.from_sampling(nb_obs=10,
                                                      margin_model=margin_model,
                                                      coordinates=coordinates)
                # Fit estimator
                estimator = SmoothMarginEstimator(dataset=dataset, margin_model=margin_model)
                estimator.fit()
                # Map name to their margin functions
                name_to_margin_function = {
                    'Ground truth margin function': dataset.margin_model.margin_function_sample,
                    'Estimated margin function': estimator.margin_function_fitted,
                }
                # Spatial Plot
                if self.DISPLAY:
                    Spatial2DPlot(name_to_margin_function=name_to_margin_function).plot()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
