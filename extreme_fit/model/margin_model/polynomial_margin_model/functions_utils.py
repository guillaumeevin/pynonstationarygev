import numpy as np

from extreme_fit.function.param_function.polynomial_coef import PolynomialCoef, PolynomialAllCoef
from extreme_fit.function.param_function.spline_coef import SplineCoef, SplineAllCoef


def load_param_name_to_polynomial_all_coef(param_name_to_list_dim_and_degree,
                                      param_name_and_dim_and_degree_to_default_coef):
    param_name_to_polynomial_all_coef = {}
    param_names = list(set([e[0] for e in param_name_and_dim_and_degree_to_default_coef.keys()]))
    for param_name in param_names:
        dim_to_polynomial_coef = {}
        for dim, max_degree in param_name_to_list_dim_and_degree.get(param_name, []):
            degree_to_coef = {}
            for (param_name_loop, dim_loop, degree), coef in param_name_and_dim_and_degree_to_default_coef.items():
                if param_name == param_name_loop and dim == dim_loop and degree <= max_degree:
                    degree_to_coef[degree] = coef
            polynomial_coef = PolynomialCoef(param_name, degree_to_coef=degree_to_coef)
            dim_to_polynomial_coef[dim] = polynomial_coef
        if len(dim_to_polynomial_coef) == 0:
            intercept = param_name_and_dim_and_degree_to_default_coef[(param_name, 0, 0)]
            dim_to_polynomial_coef = None
        else:
            intercept = None
        polynomial_all_coef = PolynomialAllCoef(param_name=param_name,
                                                dim_to_polynomial_coef=dim_to_polynomial_coef,
                                                intercept=intercept)
        param_name_to_polynomial_all_coef[param_name] = polynomial_all_coef
    return param_name_to_polynomial_all_coef


def load_param_name_to_spline_all_coef(param_name_to_list_dim_and_degree_and_nb_intervals,
                                  param_name_and_dim_and_degree_to_default_coef):
    param_name_to_spline_all_coef = {}
    param_names = list(set([e[0] for e in param_name_and_dim_and_degree_to_default_coef.keys()]))
    for param_name in param_names:
        dim_to_spline_coef = {}
        for dim, max_degree, nb_intervals in param_name_to_list_dim_and_degree_and_nb_intervals.get(param_name, []):
            nb_coefficients = nb_intervals + 1
            coefficients = np.arange(nb_coefficients)
            knots = np.arange(nb_coefficients + max_degree + 1)
            dim_to_spline_coef[dim] = SplineCoef(param_name, knots=knots, coefficients=coefficients)
        if len(dim_to_spline_coef) == 0:
            dim_to_spline_coef = None
        spline_all_coef = SplineAllCoef(param_name=param_name,
                                        dim_to_spline_coef=dim_to_spline_coef)
        param_name_to_spline_all_coef[param_name] = spline_all_coef
    return param_name_to_spline_all_coef
