from typing import Dict, Union

import numpy as np

from extreme_estimator.extreme_models.margin_model.param_function.param_function import AbstractParamFunction
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class IndependentMarginFunction(AbstractMarginFunction):
    """
        IndependentMarginFunction: each parameter of the GEV are modeled independently
    """

    def __init__(self, coordinates: AbstractCoordinates):
        """Attribute 'gev_param_name_to_param_function' maps each GEV parameter to its corresponding function"""
        super().__init__(coordinates)
        self.gev_param_name_to_param_function = None  # type: Union[None, Dict[str, AbstractParamFunction]]

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Each GEV parameter is computed independently through its corresponding param_function"""
        assert self.gev_param_name_to_param_function is not None
        assert len(self.gev_param_name_to_param_function) == 3
        transformed_coordinate = self.coordinates.transform(coordinate)
        gev_params = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            param_function = self.gev_param_name_to_param_function[gev_param_name]
            gev_params[gev_param_name] = param_function.get_gev_param_value(transformed_coordinate)
        return GevParams.from_dict(gev_params)


