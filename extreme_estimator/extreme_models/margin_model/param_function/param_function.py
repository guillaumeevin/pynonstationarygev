from typing import List
import numpy as np
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.margin_model.param_function.spline_coef import SplineCoef


class AbstractParamFunction(object):
    OUT_OF_BOUNDS_ASSERT = True

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        pass


class ConstantParamFunction(AbstractParamFunction):

    def __init__(self, constant):
        self.constant = constant

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        return self.constant


class LinearOneAxisParamFunction(AbstractParamFunction):

    def __init__(self, dim: int, coordinates: np.ndarray, coef: float = 0.01):
        self.dim = dim
        self.t_min = coordinates[:, dim].min()
        self.t_max = coordinates[:, dim].max()
        self.coef = coef

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        t = coordinate[self.dim]
        if self.OUT_OF_BOUNDS_ASSERT:
            assert self.t_min <= t <= self.t_max, '{} is out of bounds ({}, {})'.format(t, self.t_min, self.t_max)
        return t * self.coef


class LinearParamFunction(AbstractParamFunction):

    def __init__(self, dims: List[int], coordinates: np.ndarray, linear_coef: LinearCoef = None):
        self.linear_coef = linear_coef
        # Load each one axis linear function
        self.linear_one_axis_param_functions = []  # type: List[LinearOneAxisParamFunction]
        for dim in dims:
            param_function = LinearOneAxisParamFunction(dim=dim, coordinates=coordinates,
                                                        coef=self.linear_coef.get_coef(idx=dim))
            self.linear_one_axis_param_functions.append(param_function)

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        # Add the intercept and the value with respect to each axis
        gev_param_value = self.linear_coef.intercept
        for linear_one_axis_param_function in self.linear_one_axis_param_functions:
            gev_param_value += linear_one_axis_param_function.get_gev_param_value(coordinate)
        return gev_param_value


class SplineParamFunction(AbstractParamFunction):

    def __init__(self, dims, degree, spline_coef: SplineCoef, knots: np.ndarray) -> None:
        self.spline_coef = spline_coef
        self.degree = degree
        self.dims = dims
        self.knots = knots

    @property
    def m(self) -> int:
        return int((self.degree + 1) / 2)

    def get_gev_param_value(self, coordinate: np.ndarray) -> float:
        gev_param_value = self.spline_coef.intercept
        # Polynomial part
        for dim in self.dims:
            polynomial_coef = self.spline_coef.dim_to_polynomial_coef[dim]
            for degree in range(1, self.m):
                gev_param_value += polynomial_coef.get_coef(degree) * coordinate[dim]
        # Knot part
        for idx, knot in enumerate(self.knots):
            distance = np.power(np.linalg.norm(coordinate - knot), self.degree)
            gev_param_value += self.spline_coef.knot_coef.get_coef(idx) * distance
        return gev_param_value
