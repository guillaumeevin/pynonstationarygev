import unittest

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_fit.model.margin_model.spline_margin_model import Degree1SplineMarginModel
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
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
        self.margin_model = self.margin_model_class(coordinates=coordinates, params_sample={(GevParams.SHAPE, 0): 0.02})

    def test_example_visualization_2D_spatial(self):
        spatial_coordinates = LinSpaceSpatial2DCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=spatial_coordinates)

    def test_example_visualization_2D_spatio_temporal(self):
        self.nb_steps = 2
        coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]
        self.margin_model = self.margin_model_class(coordinates)


class TestVisualizationSplineMarginModel(unittest.TestCase):
    DISPLAY = False
    nb_points = 2
    margin_model_class = Degree1SplineMarginModel

    def tearDown(self) -> None:
        self.margin_model.margin_function.visualize_function(show=self.DISPLAY)
        self.assertTrue(True)

    def test_example_visualization_1D_spline(self):
        coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points, start=0.0)
        self.margin_model = self.margin_model_class(coordinates=coordinates, params_sample={(GevParams.SHAPE, 1): 0.02})

    def test_example_visualization_2D_spatial_spline(self):
        spatial_coordinates = LinSpaceSpatial2DCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=spatial_coordinates)
        # TODO: add a similar test than in the linear case
        # # Assert that the grid correspond to what we expect in a simple case
        # AbstractMarginFunction.VISUALIZATION_RESOLUTION = 2
        # grid = self.margin_model.margin_function.grid_2D['loc']
        # true_grid = np.array([[0.98, 1.0], [1.0, 1.02]])
        # self.assertTrue((grid == true_grid).all(), msg="\nexpected:\n{}, \nfound:\n{}".format(true_grid, grid))


if __name__ == '__main__':
    unittest.main()
    # v = TestVisualizationMarginModel()
    # v.test_example_visualization_2D_spatial()
    # v.tearDown()
