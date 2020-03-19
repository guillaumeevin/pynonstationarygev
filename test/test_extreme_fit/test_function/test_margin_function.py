import unittest

import numpy as np

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearAllParametersAllDimsMarginModel
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.utils import set_seed_for_test
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates, \
    LinSpaceSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from test.test_utils import load_test_spatiotemporal_coordinates, load_test_temporal_coordinates, \
    load_test_spatial_coordinates


class MarginFunction(unittest.TestCase):
    nb_points = 2
    margin_function_class = LinearMarginFunction
    margin_model_class = LinearAllParametersAllDimsMarginModel

    def test_coef_dict_spatio_temporal_coordinates(self):
        set_seed_for_test(seed=41)
        coordinates = load_test_spatiotemporal_coordinates(self.nb_points, self.nb_points)[0]
        margin_model = self.margin_model_class(coordinates)
        # Test to check loading of margin function from coef dict
        coef_dict = {'locCoeff1': 0, 'locCoeff2': 2, 'scaleCoeff1': 0,
                     'scaleCoeff2': 2, 'shapeCoeff1': 0,
                     'shapeCoeff2': 2,
                     'tempCoeffLoc1': 1, 'tempCoeffScale1': 1,
                     'tempCoeffShape1': 1}
        margin_function = self.margin_function_class.from_coef_dict(coordinates,
                                                  margin_model.margin_function_sample.gev_param_name_to_dims,
                                                  coef_dict)
        gev_param = margin_function.get_gev_params(coordinate=np.array([0.5, 1.0]), is_transformed=False)
        self.assertEqual({'loc': 2, 'scale': 2, 'shape': 2}, gev_param.to_dict())

    def test_coef_dict_spatial_coordinates(self):
        coordinates = LinSpaceSpatialCoordinates.from_nb_points(nb_points=self.nb_points+1, start=1, end=3)
        margin_model = self.margin_model_class(coordinates)
        # Test to check loading of margin function from coef dict
        coef_dict = {
            'locCoeff1': 2, 'locCoeff2': 1, 'scaleCoeff1': 0,
            'scaleCoeff2': 1, 'shapeCoeff1': 0,
            'shapeCoeff2': 1}
        margin_function = self.margin_function_class.from_coef_dict(coordinates,
                                                                    margin_model.margin_function_sample.gev_param_name_to_dims,
                                                                    coef_dict)
        gev_param = margin_function.get_gev_params(coordinate=np.array([1]), is_transformed=False)
        self.assertEqual({'loc': 3, 'scale': 1, 'shape': 1}, gev_param.to_dict())

if __name__ == '__main__':
    unittest.main()