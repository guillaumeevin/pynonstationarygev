import datetime
import time
from typing import Dict, Union

import numpy as np
import pandas as pd

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
            if self.param_name_to_climate_coordinates_with_effects is None:
                param_name_to_total_effect = None
            else:
                # Load full coordinates, and coordinates
                assert self.coordinates.nb_coordinates == 1, 'replace the line below if assert'
                temporal_coordinate = float(coordinate[0].copy())
                full_climate_coordinate = coordinate[self.coordinates.nb_coordinates:].copy()
                assert self.param_name_to_ordered_climate_effects is not None
                assert AbstractCoordinates.COORDINATE_X not in self.coordinates.coordinates_names, \
                    'check the order of coordinates that everything is ok'
                # Load full names of the effect
                param_name_to_total_effect = self.load_param_name_to_total_effect(full_climate_coordinate,
                                                                                  temporal_coordinate)
            # Update coordinate
            coordinate = np.array([float(e) for e in coordinate[:self.coordinates.nb_coordinates]])
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

    def load_param_name_to_total_effect(self, full_climate_coordinate, temporal_coordinate):
        param_name_to_total_effect = {}
        for param_name in GevParams.PARAM_NAMES:
            if self.full_climate_coordinates_names_with_effects is None:
                total_effect = 0
            elif isinstance(full_climate_coordinate[0], float):
                total_effect = self.load_total_effect_for_float(full_climate_coordinate, param_name, temporal_coordinate)
            else:
                total_effect = self.load_total_effect_for_gcm_rcm_couple(full_climate_coordinate, param_name, temporal_coordinate)
            param_name_to_total_effect[param_name] = total_effect
        return param_name_to_total_effect

    def load_total_effect_for_float(self, full_climate_coordinate, param_name, temporal_coordinate):
        assert isinstance(full_climate_coordinate[0], float)
        climate_coordinates_with_param_effects = self.param_name_to_climate_coordinates_with_effects[param_name]
        effects = self.param_name_to_ordered_climate_effects[param_name]
        if climate_coordinates_with_param_effects == self.full_climate_coordinates_names_with_effects:
            if isinstance(effects, tuple) and len(effects) == 2:
                constant_effects, linear_effects = effects
                constant_effect = np.dot(constant_effects, full_climate_coordinate)
                linear_effect = np.dot(linear_effects, full_climate_coordinate)
                linear_effect *= temporal_coordinate
                total_effect = constant_effect + linear_effect
            else:
                total_effect = np.dot(effects, full_climate_coordinate)
        elif climate_coordinates_with_param_effects is None:
            total_effect = 0
        else:
            nb_effects = len(effects)
            print('here 65', 'code goes into total effect last part')
            first_coordinate = climate_coordinates_with_param_effects[0]
            assert first_coordinate in AbstractCoordinates.COORDINATE_CLIMATE_MODEL_NAMES
            if first_coordinate == AbstractCoordinates.COORDINATE_GCM:
                total_effect = np.dot(effects, full_climate_coordinate[:nb_effects])
            elif (first_coordinate == AbstractCoordinates.COORDINATE_RCM) and (AbstractCoordinates.COORDINATE_GCM not in self.full_climate_coordinates_names_with_effects):
                total_effect = np.dot(effects, full_climate_coordinate[:nb_effects])
            else:
                total_effect = np.dot(effects, full_climate_coordinate[-nb_effects:])
        return total_effect

    def load_total_effect_for_gcm_rcm_couple(self, full_climate_coordinate, param_name, temporal_coordinate):
        # Transform the climate coordinate if they are represent with a tuple of strings
        gcm_rcm_couple = full_climate_coordinate
        column_names_for_gcm_rcm_couple = [self.coordinates.climate_model_coordinate_name_to_name_for_fit(e) for e in gcm_rcm_couple]
        if len(column_names_for_gcm_rcm_couple) == 2:
            column_names_for_gcm_rcm_couple.append(''.join(column_names_for_gcm_rcm_couple))
        column_names_for_gcm_rcm_couple.append(AbstractCoordinates.IS_ENSEMBLE_STR)
        all_column_names = self.coordinates.load_ordered_columns_names(self.full_climate_coordinates_names_with_effects)
        full_climate_coordinate = pd.Series(all_column_names).isin(column_names_for_gcm_rcm_couple).astype(float).values
        return self.load_total_effect_for_float(full_climate_coordinate, param_name, temporal_coordinate)

    def get_first_derivative_param(self, coordinate: np.ndarray, is_transformed: bool, dim: int = 0):
        transformed_coordinate = coordinate if is_transformed else self.transform(coordinate)
        return {
            param_name: param_function.get_first_derivative_param_value(transformed_coordinate, dim)
            for param_name, param_function in self.param_name_to_param_function.items()
        }
