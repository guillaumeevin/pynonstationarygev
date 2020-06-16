from typing import Dict, List

from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.polynomial_coef import PolynomialCoef


class KnotCoef(AbstractCoef):

    def __init__(self, param_name: str, default_value: float = 1.0, idx_to_coef=None):
        super().__init__(param_name, default_value, idx_to_coef)


class SplineCoef(AbstractCoef):

    def __init__(self, param_name: str, knot_coef: KnotCoef, dim_to_polynomial_coef: Dict[int, PolynomialCoef]):
        super().__init__(param_name, 1.0, None)
        self.knot_coef = knot_coef
        self.dim_to_polynomial_coef = dim_to_polynomial_coef
