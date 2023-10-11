from typing import Dict, List, Union, Tuple

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.param_function import LinearParamFunction, PolynomialParamFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class PolynomialMarginFunction(LinearMarginFunction):

    def __init__(self, coordinates: AbstractCoordinates, param_name_to_dim_and_max_degree: Dict[str, List[Tuple[int, int]]],
                 param_name_to_coef: Dict[str, PolynomialAllCoef], starting_point: Union[None, int] = None,
                 params_class: type = GevParams, log_scale=None,
                 param_name_to_ordered_climate_effects=None,
                 param_name_to_climate_coordinates_with_effects=None):
        param_name_to_dims = {}
        for param_name in param_name_to_dim_and_max_degree.keys():
            dims = [c[0] for c in param_name_to_dim_and_max_degree[param_name]]
            param_name_to_dims[param_name] = dims
        self.param_name_to_dim_and_max_degree = param_name_to_dim_and_max_degree
        super().__init__(coordinates, param_name_to_dims, param_name_to_coef, starting_point, params_class, log_scale,
                         param_name_to_ordered_climate_effects, param_name_to_climate_coordinates_with_effects)

    COEF_CLASS = PolynomialAllCoef

    def load_specific_param_function(self, param_name):
        return PolynomialParamFunction(dim_and_degree=self.param_name_to_dim_and_max_degree[param_name],
                                       coef=self.param_name_to_coef[param_name])

    def get_params(self, coordinate: np.ndarray) -> GevParams:
        return super().get_params(coordinate)

    @property
    def nb_params_for_margin_function(self):
        return sum([c.nb_params for c in self.param_name_to_coef.values()])

    @classmethod
    def from_coef_dict(cls, coordinates: AbstractCoordinates, param_name_to_dims: Dict[str, List[Tuple[int, int]]],
                       coef_dict: Dict[str, float], starting_point: Union[None, int] = None, log_scale=None,
                       param_name_to_name_of_the_climatic_effects=None, param_name_to_climate_coordinates_with_effects=None,
                       linear_effects=(False, False, False)):
        param_name_to_dim_and_max_degree = param_name_to_dims
        assert cls.COEF_CLASS is not None, 'a COEF_CLASS class attributes needs to be defined'
        param_name_to_coef = {}
        for param_name in GevParams.PARAM_NAMES:
            dims = param_name_to_dim_and_max_degree.get(param_name, [])
            coef = cls.COEF_CLASS.from_coef_dict(coef_dict=coef_dict, param_name=param_name,
                                                 dims=dims,
                                                 coordinates=coordinates)
            param_name_to_coef[param_name] = coef

        param_name_to_ordered_climate_effects = cls.load_param_name_to_ordered_climate_effects(coef_dict, param_name_to_name_of_the_climatic_effects,
                                                                                               linear_effects)

        return cls(coordinates, param_name_to_dim_and_max_degree, param_name_to_coef, starting_point, log_scale=log_scale,
                   param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects,
                   param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects)

