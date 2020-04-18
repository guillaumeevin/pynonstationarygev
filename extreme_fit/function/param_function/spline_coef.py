from typing import Dict

from extreme_fit.function.param_function.abstract_coef import AbstractCoef


class PolynomialCoef(AbstractCoef):
    """
    Object that maps each degree to its corresponding coefficient.
        degree = 1 correspond to the coefficient of the first order polynomial
        degree = 2 correspond to the the coefficient of the first order polynomial
        degree = 3 correspond to the the coefficient of the first order polynomial
    """

    def __init__(self, param_name: str, default_value: float = 1.0, degree_to_coef=None):
        super().__init__(param_name, default_value, idx_to_coef=degree_to_coef)

    def compute_default_value(self, idx):
        return self.default_value / idx


class KnotCoef(AbstractCoef):

    def __init__(self, param_name: str, default_value: float = 1.0, idx_to_coef=None):
        super().__init__(param_name, default_value, idx_to_coef)


class SplineCoef(AbstractCoef):

    def __init__(self, param_name: str, knot_coef: KnotCoef, dim_to_polynomial_coef: Dict[int, PolynomialCoef]):
        super().__init__(param_name, 1.0, None)
        self.knot_coef = knot_coef
        self.dim_to_polynomial_coef = dim_to_polynomial_coef
