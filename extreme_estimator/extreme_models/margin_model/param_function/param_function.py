from typing import List
import numpy as np
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef


class ParamFunction(object):
    OUT_OF_BOUNDS_ASSERT = True

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        pass


class ConstantParamFunction(ParamFunction):

    def __init__(self, constant):
        self.constant = constant

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        return self.constant


class LinearOneAxisParamFunction(ParamFunction):

    def __init__(self, linear_axis: int, coordinates: np.ndarray, coef: float = 0.01):
        self.linear_axis = linear_axis
        self.t_min = coordinates[:, linear_axis].min()
        self.t_max = coordinates[:, linear_axis].max()
        self.coef = coef

    def get_gev_param_value_normalized(self, coordinate: np.ndarray) -> float:
        return self.get_gev_param_value(coordinate) / (self.t_max - self.t_min)

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        t = coordinate[self.linear_axis]
        if self.OUT_OF_BOUNDS_ASSERT:
            assert self.t_min <= t <= self.t_max, 'Out of bounds'
        return t * self.coef


class LinearParamFunction(ParamFunction):

    def __init__(self, linear_dims: List[int], coordinates: np.ndarray, linear_coef: LinearCoef = None):
        self.linear_coef = linear_coef
        # Load each one axis linear function
        self.linear_one_axis_param_functions = []  # type: List[LinearOneAxisParamFunction]
        for linear_dim in linear_dims:
            param_function = LinearOneAxisParamFunction(linear_axis=linear_dim - 1, coordinates=coordinates,
                                                        coef=self.linear_coef.get_coef(dim=linear_dim))
            self.linear_one_axis_param_functions.append(param_function)

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        # Add the intercept and the value with respect to each axis
        gev_param_value = self.linear_coef.intercept
        for linear_one_axis_param_function in self.linear_one_axis_param_functions:
            gev_param_value += linear_one_axis_param_function.get_gev_param_value(coordinate)
        return gev_param_value
