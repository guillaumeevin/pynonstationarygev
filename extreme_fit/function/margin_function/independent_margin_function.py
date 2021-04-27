from typing import Dict, Union

import numpy as np

from extreme_fit.function.param_function.param_function import AbstractParamFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class IndependentMarginFunction(AbstractMarginFunction):
    """
        IndependentMarginFunction: each parameter of the GEV are modeled independently
    """

    def __init__(self, coordinates: AbstractCoordinates, params_class: type = GevParams,
                 log_scale=None, param_name_to_ordered_climate_effects=None,
                 param_name_to_climate_coordinates_with_effects=None):
        """Attribute 'param_name_to_param_function' maps each GEV parameter to its corresponding function"""
        super().__init__(coordinates, params_class, log_scale, )
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.param_name_to_ordered_climate_effects = param_name_to_ordered_climate_effects
        self.param_name_to_param_function = None  # type: Union[None, Dict[str, AbstractParamFunction]]

    @property
    def nb_params_for_climate_effects(self):
        if self.param_name_to_ordered_climate_effects is not None:
            return sum([len(effects) for effects in self.param_name_to_ordered_climate_effects.values()])
        else:
            return 0

    def get_params(self, coordinate: np.ndarray, is_transformed: bool = True) -> GevParams:
        """Each GEV parameter is computed independently through its corresponding param_function"""
        # Since all the coordinates are usually transformed by default
        # then we assume that the input coordinate are transformed by default
        assert self.param_name_to_param_function is not None
        assert len(self.param_name_to_param_function) == len(self.params_class.PARAM_NAMES)

        # Potentially separate the coordinate into two groups: the spatio temporal coordnate & the climatic coordinate
        # The climatic coordinate can be of two types either 1 and 0 vectors,
        # or a vector with several information such as the GCM str, RCM str and the climate coordinates with effects
        if len(coordinate) > self.coordinates.nb_coordinates:
            assert self.param_name_to_ordered_climate_effects is not None
            assert self.param_name_to_climate_coordinates_with_effects is not None
            assert AbstractCoordinates.COORDINATE_X not in self.coordinates.coordinates_names, \
                'check the order of coordinates that everything is ok'
            # Load full coordinates, and full names of the effect
            full_climate_coordinate = coordinate[self.coordinates.nb_coordinates:].copy()
            param_name_to_total_effect = self.load_param_name_to_total_effect(full_climate_coordinate)
            coordinate = np.array(coordinate[:self.coordinates.nb_coordinates])
        else:
            param_name_to_total_effect = None

        # Transform and compute the gev params from the param function
        assert len(coordinate) == self.coordinates.nb_coordinates
        transformed_coordinate = coordinate if is_transformed else self.transform(coordinate)
        params = {param_name: param_function.get_param_value(transformed_coordinate)
                  for param_name, param_function in self.param_name_to_param_function.items()}
        if isinstance(param_name_to_total_effect, dict):
            for param_name, total_effect in param_name_to_total_effect.items():
                params[param_name] += total_effect
        if self.log_scale:
            params[GevParams.SCALE] = np.exp(params[GevParams.SCALE])
        return self.params_class.from_dict(params)

    @property
    def full_climate_coordinates_names_with_effects(self):
        return self.coordinates.load_full_climate_coordinates_with_effects(
            self.param_name_to_climate_coordinates_with_effects)

    def load_param_name_to_total_effect(self, full_climate_coordinate):
        return {param_name: self.load_total_effect(full_climate_coordinate, param_name)
                for param_name in GevParams.PARAM_NAMES}

    def load_total_effect(self, full_climate_coordinate, param_name, climate_coordinates_names_with_param_effects_to_extract=None):
        # Load effects to consider for the parameter
        climate_coordinates_with_param_effects = self.param_name_to_climate_coordinates_with_effects[param_name]
        if climate_coordinates_names_with_param_effects_to_extract is None:
            climate_coordinates_names_with_param_effects_to_extract = climate_coordinates_with_param_effects
        assert set(climate_coordinates_names_with_param_effects_to_extract).issubset(set(climate_coordinates_with_param_effects))
        # Load param_name_to_climate_coordinate
        if isinstance(full_climate_coordinate[0], float):
            climate_coordinate = self.coordinates.get_climate_coordinate_from_full_climate_coordinate \
                (self.full_climate_coordinates_names_with_effects,
                 climate_coordinates_with_param_effects,
                 full_climate_coordinate)
        else:
            # Transform the climate coordinate if they are represent with a tuple of strings
            gcm_rcm_couple = full_climate_coordinate
            climate_coordinate = self.coordinates.get_climate_coordinate_from_gcm_rcm_couple \
                (self.full_climate_coordinates_names_with_effects, climate_coordinates_with_param_effects,
                 gcm_rcm_couple, climate_coordinates_names_with_param_effects_to_extract)
        # Compute total effect
        effects = self.param_name_to_ordered_climate_effects[param_name]
        assert len(effects) == len(climate_coordinate)
        return np.dot(effects, climate_coordinate)

    def get_first_derivative_param(self, coordinate: np.ndarray, is_transformed: bool, dim: int = 0):
        transformed_coordinate = coordinate if is_transformed else self.transform(coordinate)
        return {
            param_name: param_function.get_first_derivative_param_value(transformed_coordinate, dim)
            for param_name, param_function in self.param_name_to_param_function.items()
        }
