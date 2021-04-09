from typing import Dict, List

from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.polynomial_coef import PolynomialCoef

from typing import Dict, List

from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class SplineCoef(AbstractCoef):

    def __init__(self, param_name: str, coefficients, knots):
        super().__init__(param_name)
        self.knots = knots
        self.coefficients = coefficients
        self.max_degree = self.nb_knots - (self.nb_coefficients + 1)

    @property
    def nb_knots(self):
        return len(self.knots)

    @property
    def nb_coefficients(self):
        return len(self.coefficients)

    @property
    def nb_params(self):
        return self.nb_knots + self.nb_coefficients

class SplineAllCoef(LinearCoef):

    def __init__(self, param_name, dim_to_spline_coef: Dict[int, SplineCoef]):
        super().__init__(param_name, 1.0, None)
        self.dim_to_spline_coef = dim_to_spline_coef

    def form_dict(self, coordinates_names: List[str], dims) -> Dict[str, str]:
        formula_list = []
        if len(coordinates_names) == 0:
            formula_str = '1'
        else:
            for dim, name in zip(dims, coordinates_names):
                spline_coef = self.dim_to_spline_coef[dim]
                formula_list.append('s({}, m={}, k={}, bs="bs")'.format(name, spline_coef.max_degree,
                                                                        spline_coef.nb_coefficients))
            formula_str = ' '.join(formula_list)
        return {self.param_name + '.form': self.param_name + ' ~ ' + formula_str}

    @property
    def nb_params(self):
        return sum([spline_coef.nb_params for spline_coef in self.dim_to_spline_coef.values()])