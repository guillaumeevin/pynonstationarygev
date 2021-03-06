import operator
from functools import reduce
from typing import List
import numpy as np
from scipy.interpolate import BSpline

from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.one_axis_param_function import LinearOneAxisParamFunction
from extreme_fit.function.param_function.polynomial_coef import PolynomialAllCoef, PolynomialCoef
from extreme_fit.function.param_function.spline_coef import PolynomialCoef, SplineAllCoef


class AbstractParamFunction(object):
    OUT_OF_BOUNDS_ASSERT = True

    def get_param_value(self, coordinate: np.ndarray) -> float:
        raise NotImplementedError

    def get_first_derivative_param_value(self, coordinate: np.ndarray, dim: int) -> float:
        raise NotImplementedError


class ConstantParamFunction(AbstractParamFunction):

    def __init__(self, constant):
        self.constant = constant

    def get_param_value(self, coordinate: np.ndarray) -> float:
        return self.constant

    def get_first_derivative_param_value(self, coordinate: np.ndarray, dim: int) -> float:
        return 0


class LinearParamFunction(AbstractParamFunction):

    def __init__(self, dims: List[int], coordinates: np.ndarray, linear_coef: LinearCoef):
        self.linear_coef = linear_coef
        # Load each one axis linear function
        self.linear_one_axis_param_functions = []  # type: List[LinearOneAxisParamFunction]
        for dim in dims:
            param_function = LinearOneAxisParamFunction(dim=dim, coordinates=coordinates,
                                                        coef=self.linear_coef.get_coef(idx=dim))
            self.linear_one_axis_param_functions.append(param_function)
        self.dim_to_linear_one_axis_param_function = dict(zip(dims, self.linear_one_axis_param_functions))

    def get_param_value(self, coordinate: np.ndarray) -> float:
        # Add the intercept and the value with respect to each axis
        gev_param_value = self.linear_coef.intercept
        for linear_one_axis_param_function in self.linear_one_axis_param_functions:
            gev_param_value += linear_one_axis_param_function.get_param_value(coordinate, self.OUT_OF_BOUNDS_ASSERT)
        return gev_param_value

    def get_first_derivative_param_value(self, coordinate: np.ndarray, dim: int) -> float:
        return self.dim_to_linear_one_axis_param_function[dim].coef


class PolynomialParamFunction(AbstractParamFunction):

    def __init__(self, dim_and_degree, coef: PolynomialAllCoef):
        self.coef = coef
        self.dim_and_degree = dim_and_degree

    def get_param_value(self, coordinate: np.ndarray) -> float:
        gev_param_value = 0
        for i, (dim, max_degree) in enumerate(self.dim_and_degree):
            if isinstance(dim, int):
                coordinate_value_for_dim = coordinate[dim]
            else:
                coordinate_value_for_dim = reduce(operator.mul, [coordinate[d] for d in dim])
            # Add intercept only once
            add_intercept = i == 0
            first_degree = 0 if add_intercept else 1
            for degree in range(first_degree, max_degree+1):
                polynomial_coef = self.coef.dim_to_polynomial_coef[dim]  # type: PolynomialCoef
                polynomial_coef_value = polynomial_coef.idx_to_coef[degree]
                gev_param_value += polynomial_coef_value * np.power(coordinate_value_for_dim, degree)
        return gev_param_value


class SplineParamFunction(AbstractParamFunction):

    def __init__(self, dim_and_degree, coef: SplineAllCoef) -> None:
        self.coef = coef
        self.dim_and_degree = dim_and_degree

    def get_param_value(self, coordinate: np.ndarray) -> float:
        gev_param_value = 0
        for i, (dim, max_degree) in enumerate(self.dim_and_degree):
            assert isinstance(dim, int)
            spline_coef = self.coef.dim_to_spline_coef[dim]
            spl = BSpline(t=spline_coef.knots, c=spline_coef.coefficients, k=max_degree, extrapolate=False)
            gev_param_value += spl(coordinate[dim])
        return gev_param_value
