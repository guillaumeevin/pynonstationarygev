from typing import Dict, List, Union, Tuple

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.function.margin_function.polynomial_margin_function import PolynomialMarginFunction
from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.param_function import LinearParamFunction, PolynomialParamFunction, \
    SplineParamFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef
from extreme_fit.function.param_function.spline_coef import SplineAllCoef, SplineCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class SplineMarginFunction(LinearMarginFunction):

    def __init__(self, coordinates: AbstractCoordinates,
                 param_name_to_dim_and_max_degree: Dict[str, List[Tuple[int, int]]],
                 param_name_to_coef: Dict[str, SplineAllCoef], starting_point: Union[None, int] = None,
                 params_class: type = GevParams, log_scale=None,
                 param_name_to_ordered_climate_effects=None):
        param_name_to_dims = {}
        for param_name in param_name_to_dim_and_max_degree.keys():
            dims = [c[0] for c in param_name_to_dim_and_max_degree[param_name]]
            param_name_to_dims[param_name] = dims
        self.param_name_to_dim_and_max_degree = param_name_to_dim_and_max_degree
        super().__init__(coordinates, param_name_to_dims, param_name_to_coef, starting_point, params_class, log_scale,
                         param_name_to_ordered_climate_effects)

    COEF_CLASS = SplineAllCoef

    def load_specific_param_function(self, param_name):
        coef = self.param_name_to_coef[param_name]
        assert isinstance(coef, (PolynomialAllCoef, SplineAllCoef))
        if isinstance(coef, PolynomialAllCoef):
            return PolynomialParamFunction(dim_and_degree=self.param_name_to_dim_and_max_degree[param_name], coef=coef)
        else:
            return SplineParamFunction(dim_and_degree=self.param_name_to_dim_and_max_degree[param_name], coef=coef)

    def get_params(self, coordinate: np.ndarray, is_transformed: bool = True) -> GevParams:
        return super().get_params(coordinate, is_transformed)

    @property
    def nb_params_for_margin_function(self):
        return sum([c.nb_params for c in self.param_name_to_coef.values()])

    @classmethod
    def from_coef_dict(cls, coordinates: AbstractCoordinates, param_name_to_dims: Dict[str, List[Tuple[int, int]]],
                       coef_dict: Dict[str, float], starting_point: Union[None, int] = None, log_scale=None,
                       param_name_to_name_of_the_climatic_effects=None, param_name_to_climate_coordinates_with_effects=None):

        coef_dict, spline_param_name_to_dim_to_knots_and_coefficient = coef_dict
        # Load polynomial coefficient
        polynomial_margin_function = PolynomialMarginFunction.from_coef_dict(coordinates, param_name_to_dims, coef_dict,
                                                                             starting_point, log_scale,
                                                                             param_name_to_name_of_the_climatic_effects)
        param_name_to_coef = polynomial_margin_function.param_name_to_coef
        param_name_to_dim_and_max_degree = param_name_to_dims
        # Load the remaining spline coefficient
        assert cls.COEF_CLASS is not None, 'a COEF_CLASS class attributes needs to be defined'
        for param_name, dim_to_knots_and_coefficients in spline_param_name_to_dim_to_knots_and_coefficient.items():
            dim_to_spline_coef = {}
            for dim, (knots, coefficients) in dim_to_knots_and_coefficients.items():
                dim_to_spline_coef[dim] = SplineCoef(param_name, coefficients, knots)
            param_name_to_coef[param_name] = SplineAllCoef(param_name, dim_to_spline_coef)

        return cls(coordinates, param_name_to_dim_and_max_degree, param_name_to_coef, starting_point, log_scale=log_scale,
                   param_name_to_ordered_climate_effects=polynomial_margin_function.param_name_to_ordered_climate_effects,
                   param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects)
