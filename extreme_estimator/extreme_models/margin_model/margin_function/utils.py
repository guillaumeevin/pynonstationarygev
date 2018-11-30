import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.gev_params import GevParams


def abs_error(s1, s2):
    return (s1 - s2).abs().pow(2)


def error_dict_between_margin_functions(margin1: AbstractMarginFunction, margin2: AbstractMarginFunction):
    assert margin1.coordinates == margin2.coordinates
    margin1_gev_params, margin2_gev_params = margin1.gev_params_for_coordinates, margin2.gev_params_for_coordinates
    gev_param_name_to_error_serie = {}
    for gev_param_name in GevParams.GEV_PARAM_NAMES:
        serie1, serie2 = margin1_gev_params[gev_param_name], margin2_gev_params[gev_param_name]
        error = abs_error(serie1, serie2)
        gev_param_name_to_error_serie[gev_param_name] = error
    return gev_param_name_to_error_serie
