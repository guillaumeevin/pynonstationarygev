import unittest

from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel, \
    LinearShapeAxis0MarginModel, LinearShapeAxis0and1MarginModel, LinearAllParametersAxis0MarginModel, \
    LinearAllParametersAxis0And1MarginModel
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.return_level_plot.spatial_2D_plot import Spatial2DPlot
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from test.test_utils import load_smooth_margin_models


class TestSmoothMarginEstimator(unittest.TestCase):
    DISPLAY = False
    SMOOTH_MARGIN_ESTIMATORS = [SmoothMarginEstimator]

    def setUp(self):
        super().setUp()
        self.spatial_coordinates = CircleCoordinates.from_nb_points(nb_points=5, max_radius=1)
        self.smooth_margin_models = load_smooth_margin_models(coordinates=self.spatial_coordinates)

    def test_dependency_estimators(self):
        for margin_model in self.smooth_margin_models:
            dataset = MarginDataset.from_sampling(nb_obs=10, margin_model=margin_model,
                                                  coordinates=self.spatial_coordinates)
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
