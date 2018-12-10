import unittest

from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeDim1MarginModel, \
    LinearAllParametersAllDimsMarginModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates


class VisualizationMarginModel(unittest.TestCase):
    DISPLAY = False
    nb_points = 2
    margin_model = [LinearShapeDim1MarginModel, LinearAllParametersAllDimsMarginModel][-1]

    def test_example_visualization_2D(self):
        spatial_coordinates = CircleSpatialCoordinates.from_nb_points(nb_points=self.nb_points)
        margin_model = self.margin_model(coordinates=spatial_coordinates)
        if self.DISPLAY:
            margin_model.margin_function_sample.visualize()

    def test_example_visualization_1D(self):
        coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points)
        margin_model = self.margin_model(coordinates=coordinates, params_sample={(GevParams.GEV_SHAPE, 1): 0.02})
        margin_model.margin_function_sample.visualize(show=self.DISPLAY)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
