from typing import Dict

import numpy as np

from extreme_estimator.extreme_models.margin_model.param_function.param_function import ParamFunction
from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class IndependentMarginFunction(AbstractMarginFunction):
    """Margin Function where each parameter of the GEV are modeled independently"""

    def __init__(self, coordinates: AbstractCoordinates):
        """Attribute 'gev_param_name_to_param_function' maps each GEV parameter to its corresponding function"""
        super().__init__(coordinates)
        self.gev_param_name_to_param_function = None  # type: Dict[str, ParamFunction]

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Each GEV parameter is computed independently through its corresponding param_function"""
        assert self.gev_param_name_to_param_function is not None
        assert len(self.gev_param_name_to_param_function) == 3
        gev_params = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            param_function = self.gev_param_name_to_param_function[gev_param_name]
            gev_params[gev_param_name] = param_function.get_gev_param_value(coordinate)
        return GevParams.from_dict(gev_params)


