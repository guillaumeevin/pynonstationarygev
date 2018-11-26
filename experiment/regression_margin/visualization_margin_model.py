import unittest

from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeAxis0MarginModel, \
    LinearAllParametersAllAxisMarginModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import LinSpaceCoordinates


class VisualizationMarginModel(unittest.TestCase):
    DISPLAY = True
    nb_points = 50
    margin_model = [LinearShapeAxis0MarginModel, LinearAllParametersAllAxisMarginModel][-1]

    @classmethod
    def example_visualization_2D(cls):
        spatial_coordinates = CircleCoordinates.from_nb_points(nb_points=cls.nb_points)
        margin_model = cls.margin_model(coordinates=spatial_coordinates)
        if cls.DISPLAY:
            margin_model.margin_function_sample.visualize_all()

    @classmethod
    def example_visualization_1D(cls):
        coordinates = LinSpaceCoordinates.from_nb_points(nb_points=cls.nb_points)
        # MarginModel Linear with respect to the shape (from 0.01 to 0.02)
        margin_model = cls.margin_model(coordinates=coordinates, params_sample={GevParams.GEV_SHAPE: 0.02})
        if cls.DISPLAY:
            margin_model.margin_function_sample.visualize_all()


if __name__ == '__main__':
    VisualizationMarginModel.example_visualization_1D()
    VisualizationMarginModel.example_visualization_2D()
