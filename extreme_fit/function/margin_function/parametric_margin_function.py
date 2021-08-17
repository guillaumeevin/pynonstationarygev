from typing import Dict, List, Union

import numpy as np

from extreme_fit.function.margin_function.independent_margin_function import \
    IndependentMarginFunction
from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.param_function import AbstractParamFunction, \
    ConstantParamFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ParametricMarginFunction(IndependentMarginFunction):
    """
    ParametricMarginFunction each parameter of the GEV will:

        -depend on some integer dimensions (dimension 1 or/and dimension 2 for instance).
        Coordinate name corresponding to the dimension depends on the order of the columns of self.coordinates
        param_name_to_dims maps each GEV parameter to its corresponding dimensions

        -have a set of all potential coefficient that could be used to define a function
        param_name_to_coef maps each GEV parameter to an AbstractCoef object. This object contains

        -combining the integer dimensions & the set of all potential coefficient
        to keep only the relevant coefficient, and build the corresponding function from that
        param_name_to_param_function maps each GEV parameter to a AbstractParamFunction object.

    """

    COEF_CLASS = None

    def __init__(self, coordinates: AbstractCoordinates, param_name_to_dims: Dict[str, List[int]],
                 param_name_to_coef: Dict[str, AbstractCoef], starting_point: Union[None, int] = None,
                 params_class: type = GevParams,
                 log_scale=None,
                 param_name_to_ordered_climate_effects=None,
                 param_name_to_climate_coordinates_with_effects=None):
        # Starting point for the trend is the same for all the parameters
        self.starting_point = starting_point
        super().__init__(coordinates, params_class, log_scale=log_scale,
                         param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects,
                         param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects)
        self.param_name_to_dims = param_name_to_dims  # type: Dict[str, List[int]]

        # Check the dimension are well-defined with respect to the coordinates
        for dims in self.param_name_to_dims.values():
            for dim in dims:
                if isinstance(dim, int):
                    assert 0 <= dim < coordinates.nb_coordinates, \
                        "dim={}, nb_columns={}".format(dim, coordinates.nb_coordinates)
                elif isinstance(dim, tuple):
                    for d in dim:
                        assert 0 <= d < coordinates.nb_coordinates, \
                            "dim={}, nb_columns={}".format(d, coordinates.nb_coordinates)
                else:
                    raise TypeError(type(dim))

        self.param_name_to_coef = param_name_to_coef  # type: Dict[str, AbstractCoef]

        # Build gev_parameter_to_param_function dictionary
        self.param_name_to_param_function = {}  # type: Dict[str, AbstractParamFunction]
        # Map each param_name to its corresponding param_function
        for param_name in self.params_class.PARAM_NAMES:
            # By default, if dims are not specified, a constantParamFunction is chosen
            if self.param_name_to_dims.get(param_name) is None:
                param_function = ConstantParamFunction(constant=self.param_name_to_coef[param_name].intercept)
            # Otherwise, we load a specific param function
            else:
                param_function = self.load_specific_param_function(param_name)
            # In both cases, we add the param_function to the dictionary
            self.param_name_to_param_function[param_name] = param_function

    def load_specific_param_function(self, param_name) -> AbstractParamFunction:
        raise NotImplementedError

    @property
    def transformed_starting_point(self):
        return self.coordinates.temporal_coordinates.transformation.transform_array(np.array([self.starting_point]))

    def get_params(self, coordinate: np.ndarray, is_transformed: bool = True) -> GevParams:
        coordinate = self.shift_coordinates_if_needed(coordinate, is_transformed)
        return super().get_params(coordinate, is_transformed=is_transformed)

    def shift_coordinates_if_needed(self, coordinate, is_transformed):
        if self.starting_point is not None:
            starting_point = self.transformed_starting_point if is_transformed else self.starting_point
            # Shift temporal coordinate to enable to model temporal trend with starting point
            assert self.coordinates.has_temporal_coordinates
            assert 0 <= self.coordinates.idx_temporal_coordinates < len(coordinate)
            if coordinate[self.coordinates.idx_temporal_coordinates] < starting_point:
                coordinate[self.coordinates.idx_temporal_coordinates] = starting_point
        return coordinate

    @classmethod
    def from_coef_dict(cls, coordinates: AbstractCoordinates, param_name_to_dims: Dict[str, List[int]],
                       coef_dict: Dict[str, float], starting_point: Union[None, int] = None,
                       log_scale=None, param_name_to_name_of_the_climatic_effects=None,
                       param_name_to_climate_coordinates_with_effects=None,
                       param_name_to_ordered_climate_effects=None,
                       linear_effects=False):
        assert cls.COEF_CLASS is not None, 'a COEF_CLASS class attributes needs to be defined'
        # Load param_name_to_coef
        param_name_to_coef = {}
        for param_name in GevParams.PARAM_NAMES:
            dims = param_name_to_dims.get(param_name, [])
            coef = cls.COEF_CLASS.from_coef_dict(coef_dict=coef_dict, param_name=param_name, dims=dims,
                                                 coordinates=coordinates)
            param_name_to_coef[param_name] = coef
        # Load param_name_to_ordered_climate_effects
        if param_name_to_ordered_climate_effects is None:
            param_name_to_ordered_climate_effects = cls.load_param_name_to_ordered_climate_effects(coef_dict,
                                                                                                   param_name_to_name_of_the_climatic_effects,
                                                                                                   linear_effects)
        return cls(coordinates, param_name_to_dims, param_name_to_coef,
                   starting_point=starting_point, log_scale=log_scale,
                   param_name_to_ordered_climate_effects=param_name_to_ordered_climate_effects,
                   param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects)

    @classmethod
    def load_param_name_to_ordered_climate_effects(cls, coef_dict, param_name_to_name_of_the_climatic_effects,
                                                   linear_effects):
        if param_name_to_name_of_the_climatic_effects is None:
            param_name_to_ordered_climate_effects = None
        else:
            param_name_to_ordered_climate_effects = {}
            for param_name in GevParams.PARAM_NAMES:
                names = param_name_to_name_of_the_climatic_effects[param_name]
                ordered_climate_effects = [coef_dict[param_name + name] for name in names]
                if linear_effects:
                    ordered_climate_effects_linear = [coef_dict[param_name + name + AbstractCoordinates.COORDINATE_T] for name in names]
                    param_name_to_ordered_climate_effects[param_name] = (ordered_climate_effects, ordered_climate_effects_linear)
                else:
                    param_name_to_ordered_climate_effects[param_name] = ordered_climate_effects
        return param_name_to_ordered_climate_effects

    @property
    def form_dict(self) -> Dict[str, str]:
        raise NotImplementedError

    @property
    def coef_dict(self) -> Dict[str, str]:
        raise NotImplementedError
