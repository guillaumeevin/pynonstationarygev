from typing import Dict, List

from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class PolynomialCoef(AbstractCoef):
    """
    Object that maps each degree to its corresponding coefficient.
        degree = 1 correspond to the coefficient of the first order polynomial
        degree = 2 correspond to the the coefficient of the first order polynomial
        degree = 3 correspond to the the coefficient of the first order polynomial
    """

    def __init__(self, param_name: str, default_value: float = 1.0, degree_to_coef=None):
        super().__init__(param_name, default_value, idx_to_coef=degree_to_coef)
        self.max_degree = max(self.idx_to_coef.keys()) if self.idx_to_coef is not None else None

    def compute_default_value(self, idx):
        return self.default_value / idx

    @property
    def nb_params(self):
        return self.max_degree + 1


class PolynomialAllCoef(LinearCoef):

    def __init__(self, param_name, dim_to_polynomial_coef: Dict[int, PolynomialCoef], intercept=None):
        super().__init__(param_name, 1.0, None)
        self.dim_to_polynomial_coef = dim_to_polynomial_coef
        self._intercept = intercept

    @property
    def nb_params(self):
        if self.dim_to_polynomial_coef is None:
            return 1
        else:
            return sum([c.nb_params for c in self.dim_to_polynomial_coef.values()])

    @property
    def intercept(self) -> float:
        if self._intercept is not None:
            return self._intercept
        else:
            return super().intercept

    @classmethod
    def from_coef_dict(cls, coef_dict: Dict[str, float], param_name: str, dims: List[int],
                       coordinates: AbstractCoordinates):
        degree0 = coef_dict[cls.coef_template_str(param_name, coefficient_name=cls.INTERCEPT_NAME).format(1)]
        list_dim_and_max_degree = dims
        j = 2
        if len(list_dim_and_max_degree) == 0:
            dim_to_polynomial_coef = None
            intercept = degree0
        else:
            intercept = None
            dim_to_polynomial_coef = {}
            for dim, max_degree in list_dim_and_max_degree:
                coefficient_name = coordinates.coordinates_names[dim]
                if coefficient_name == AbstractCoordinates.COORDINATE_T:
                    j = 1
                degree_to_coef = {0: degree0}
                for degree in range(1, max_degree + 1):
                    coef_value = coef_dict[cls.coef_template_str(param_name, coefficient_name).format(j)]
                    degree_to_coef[degree] = coef_value
                    j += 1
                dim_to_polynomial_coef[dim] = PolynomialCoef(param_name=param_name, degree_to_coef=degree_to_coef)
        return cls(param_name=param_name, dim_to_polynomial_coef=dim_to_polynomial_coef, intercept=intercept)

    def form_dict(self, coordinates_names: List[str], dims) -> Dict[str, str]:
        if len(coordinates_names) >= 2:
            raise NotImplementedError(
                'Check how do we sum two polynomails without having two times an intercept parameter')
        formula_list = []
        if len(coordinates_names) == 0:
            formula_str = '1'
        else:
            for dim, name in zip(dims, coordinates_names):
                polynomial_coef = self.dim_to_polynomial_coef[dim]
                formula_list.append('poly({}, {}, raw = TRUE)'.format(name, polynomial_coef.max_degree))
            formula_str = ' '.join(formula_list)
        return {self.param_name + '.form': self.param_name + ' ~ ' + formula_str}
