import numpy as np
import unittest

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearShapeDim1MarginModel, \
    LinearAllParametersAllDimsMarginModel
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import \
    CircleSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates
from test.test_utils import load_test_spatiotemporal_coordinates


class TestVisualizationMarginModel(unittest.TestCase):
    DISPLAY = False
    nb_points = 2
    margin_model_class = [LinearShapeDim1MarginModel, LinearAllParametersAllDimsMarginModel][-1]

    def tearDown(self) -> None:
        self.margin_model.margin_function_sample.visualize_function(show=self.DISPLAY)
        self.assertTrue(True)

    def test_example_visualization_1D(self):
        coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=coordinates, params_sample={(GevParams.SHAPE, 1): 0.02})

    def test_example_visualization_2D_spatial(self):
        spatial_coordinates = LinSpaceSpatial2DCoordinates.from_nb_points(nb_points=self.nb_points)
        self.margin_model = self.margin_model_class(coordinates=spatial_coordinates)
        # Assert that the grid correspond to what we expect in a simple case
        AbstractMarginFunction.VISUALIZATION_RESOLUTION = 2
        grid = self.margin_model.margin_function_sample.grid_2D['loc']
        true_grid = np.array([[0.98, 1.0], [1.0, 1.02]])
        self.assertTrue((grid == true_grid).all(), msg="\nexpected:\n{}, \nfound:\n{}".format(true_grid, grid))

    # def test_example_visualization_2D_spatio_temporal(self):
    #     self.nb_steps = 2
    #     coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[0]
    #     self.margin_model = self.margin_model_class(coordinates)
    #
    #     # Load margin function from coef dict
    #     coef_dict = {'locCoeff1': 0, 'locCoeff2': 1, 'scaleCoeff1': 0,
    #                  'scaleCoeff2': 1, 'shapeCoeff1': 0,
    #                  'shapeCoeff2': 1,
    #                  'tempCoeffLoc1': 1, 'tempCoeffScale1': 1,
    #                  'tempCoeffShape1': 1}
    #     margin_function = LinearMarginFunction.from_coef_dict(coordinates,
    #                                                           self.margin_model.margin_function_sample.gev_param_name_to_linear_dims,
    #                                                           coef_dict)
    #     self.margin_model.margin_function_sample = margin_function
    #     self.margin_model.margin_function_sample.visualize_2D(show=True)
    #
    #     # Load


if __name__ == '__main__':
    unittest.main()
    # v = TestVisualizationMarginModel()
    # v.test_example_visualization_2D_spatial()
    # v.tearDown()
