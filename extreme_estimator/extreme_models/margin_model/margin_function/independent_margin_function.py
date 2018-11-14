from typing import Dict

import numpy as np

from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import AbstractMarginFunction
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class ParamFunction(object):

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        pass


class IndependentMarginFunction(AbstractMarginFunction):

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams):
        super().__init__(spatial_coordinates, default_params)
        self.gev_param_name_to_param_function = None  # type: Dict[str, ParamFunction]

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        assert self.gev_param_name_to_param_function is not None
        assert len(self.gev_param_name_to_param_function) == 3
        gev_params = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            param_function = self.gev_param_name_to_param_function[gev_param_name]
            gev_value = param_function.get_gev_param_value(coordinate)
            gev_params[gev_param_name] = gev_value
        return GevParams.from_dict(gev_params)


class ConstantParamFunction(ParamFunction):

    def __init__(self, constant):
        self.constant = constant

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        return self.constant


class LinearOneAxisParamFunction(ParamFunction):

    def __init__(self, linear_axis, coordinates_axis, start, end=0.0):
        self.linear_axis = linear_axis
        self.t_min = coordinates_axis.min()
        self.t_max = coordinates_axis.max()
        self.start = start
        self.end = end

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        t = coordinate[self.linear_axis]
        t_between_zero_and_one = (t - self.t_min) / self.t_max
        return self.start + t_between_zero_and_one * (self.end - self.start)


class LinearMarginFunction(IndependentMarginFunction):
    """
    On the minimal point along all the dimension, the GevParms will equal default params
    Otherwise, it will augment linearly along a single linear axis
    """

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams,
                 gev_param_name_to_linear_axis: Dict[str, int]):
        super().__init__(spatial_coordinates, default_params)
        self.param_to_linear_dims = gev_param_name_to_linear_axis
        assert all([axis < np.ndim(spatial_coordinates.coordinates) for axis in gev_param_name_to_linear_axis.values()])
        # Initialize gev_parameter_to_param_function
        self.gev_param_name_to_param_function = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            if gev_param_name not in gev_param_name_to_linear_axis.keys():
                param_function = ConstantParamFunction(constant=self.default_params[gev_param_name])
            else:
                linear_axis = gev_param_name_to_linear_axis.get(gev_param_name, None)
                coordinates_axis = self.spatial_coordinates.coordinates[:, linear_axis]
                param_function = LinearOneAxisParamFunction(linear_axis=linear_axis, coordinates_axis=coordinates_axis,
                                                            start=self.default_params[gev_param_name])
            self.gev_param_name_to_param_function[gev_param_name] = param_function
