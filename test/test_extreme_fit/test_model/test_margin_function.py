import unittest

import numpy as np

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from test.test_utils import load_test_spatiotemporal_coordinates


class MarginFunction(unittest.TestCase):
    nb_points = 2
    margin_function_class = LinearMarginFunction
    margin_model_class = LinearAllParametersAllDimsMarginModel

    def test_grid_2D_orientation(self):
        # Assert that the grid correspond to what we expect in a simple case
        margin_model = self.margin_model_class(LinSpaceSpatial2DCoordinates.from_nb_points(nb_points=self.nb_points))
        AbstractMarginFunction.VISUALIZATION_RESOLUTION = 2
        grid = margin_model.margin_function_sample.grid_2D()['loc']
        true_grid = np.array([[0.98, 1.0], [1.0, 1.02]])
        self.assertTrue((grid == true_grid).all(), msg="\nexpected:\n{}, \nfound:\n{}".format(true_grid, grid))

    def test_coef_dict(self):
        coordinates = load_test_spatiotemporal_coordinates(self.nb_points, self.nb_points)[0]
        margin_model = self.margin_model_class(coordinates)
        # Test to check loading of margin function from coef dict
        coef_dict = {'locCoeff1': 0, 'locCoeff2': 1, 'scaleCoeff1': 0,
                     'scaleCoeff2': 1, 'shapeCoeff1': 0,
                     'shapeCoeff2': 1,
                     'tempCoeffLoc1': 1, 'tempCoeffScale1': 1,
                     'tempCoeffShape1': 1}
        self.margin_function_class.from_coef_dict(coordinates,
                                                  margin_model.margin_function_sample.gev_param_name_to_dims,
                                                  coef_dict)
