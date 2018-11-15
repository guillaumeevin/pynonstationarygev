import unittest

from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel, \
    LinearShapeAxis0MarginModel
from extreme_estimator.estimator.margin_estimator import SmoothMarginEstimator
from extreme_estimator.return_level_plot.spatial_2D_plot import Spatial2DPlot
from spatio_temporal_dataset.dataset.simulation_dataset import MarginDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestSmoothMarginEstimator(unittest.TestCase):
    DISPLAY = False
    MARGIN_TYPES = [ConstantMarginModel, LinearShapeAxis0MarginModel][1:]
    MARGIN_ESTIMATORS = [SmoothMarginEstimator]

    def setUp(self):
        super().setUp()
        self.spatial_coordinates = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.margin_models = self.load_margin_models(spatial_coordinates=self.spatial_coordinates)

    @classmethod
    def load_margin_models(cls, spatial_coordinates):
        return [margin_class(spatial_coordinates=spatial_coordinates) for margin_class in cls.MARGIN_TYPES]

    def test_dependency_estimators(self):
        for margin_model in self.margin_models:
            dataset = MarginDataset.from_sampling(nb_obs=10, margin_model=margin_model,
                                                  spatial_coordinates=self.spatial_coordinates)
            # Fit estimator
            estimator = SmoothMarginEstimator(dataset=dataset, margin_model=margin_model)
            estimator.fit()
            # Map name to their margin functions
            name_to_margin_function = {
                'Ground truth margin function': dataset.margin_model.margin_function_sample,
                # 'Estimated margin function': estimator.margin_function_fitted,
            }
            # Spatial Plot
            if self.DISPLAY:
                Spatial2DPlot(name_to_margin_function=name_to_margin_function).plot()
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
