import unittest

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import NonStationaryTwoLinearLocationModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from test.test_utils import load_test_spatiotemporal_coordinates


class TestVisualizationLinearMarginModel(unittest.TestCase):
    DISPLAY = False
    nb_points = 2
    margin_model_class = LinearAllParametersAllDimsMarginModel

    def tearDown(self) -> None:
        self.margin_model.margin_function.visualize_function(show=self.DISPLAY)
        self.assertTrue(True)

    def test_example_visualization_1D(self):
        coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=coordinates, params_user={(GevParams.SHAPE, 0): 0.02})

    def test_example_visualization_2D_spatial(self):
        spatial_coordinates = LinSpaceSpatial2DCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=spatial_coordinates)

    def test_example_visualization_2D_spatio_temporal(self):
        self.nb_steps = 2
        coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]
        self.margin_model = self.margin_model_class(coordinates)


class TestVisualizationSplineMarginModel(unittest.TestCase):
    DISPLAY = True
    nb_points = 50
    margin_model_class = NonStationaryTwoLinearLocationModel

    # def tearDown(self) -> None:
    #     self.margin_model.margin_function.visualize_function(show=self.DISPLAY)
    #     self.assertTrue(True)

    # def test_example_visualization_1D_spline(self):
    #     coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=self.nb_points)
    #     self.margin_model = self.margin_model_class(coordinates=coordinates)


if __name__ == '__main__':
    unittest.main()
    # v = TestVisualizationMarginModel()
    # v.test_example_visualization_2D_spatial()
    # v.tearDown()
