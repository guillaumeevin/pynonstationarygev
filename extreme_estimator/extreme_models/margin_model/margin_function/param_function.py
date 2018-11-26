from typing import List

import numpy as np


class ParamFunction(object):

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        pass


class ConstantParamFunction(ParamFunction):

    def __init__(self, constant):
        self.constant = constant

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        return self.constant


class LinearOneAxisParamFunction(ParamFunction):

    def __init__(self, linear_axis: int, coordinates_axis: np.ndarray, start: float, end: float = 0.01):
        self.linear_axis = linear_axis
        self.t_min = coordinates_axis.min()
        self.t_max = coordinates_axis.max()
        self.start = start
        self.end = end

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        t = coordinate[self.linear_axis]
        t_between_zero_and_one = (t - self.t_min) / (self.t_max - self.t_min)
        assert 0 <= t_between_zero_and_one <= 1, 'Out of bounds'
        return self.start + t_between_zero_and_one * (self.end - self.start)


class LinearParamFunction(ParamFunction):

    def __init__(self, linear_axes: List[int], coordinates: np.ndarray, start: float, end: float = 0.01):
        self.linear_one_axis_param_functions = []  # type: List[LinearOneAxisParamFunction]
        self.start = start
        self.end = end
        for linear_axis in linear_axes:
            param_function = LinearOneAxisParamFunction(linear_axis, coordinates[:, linear_axis], start, end)
            self.linear_one_axis_param_functions.append(param_function)

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        values = [param_funct.get_gev_param_value(coordinate) for param_funct in self.linear_one_axis_param_functions]
        return float(np.mean(values))
