import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.gev_params import GevParams


def relative_abs_error(reference_value, fitted_value):
    return (reference_value - fitted_value).abs() / reference_value


def error_dict_between_margin_functions(reference: AbstractMarginFunction, fitted: AbstractMarginFunction):
    """
    Return a serie, indexed by the same index as the coordinates
    Each value correspond to the error for this coordinate
    :param reference:
    :param fitted:
    :return:
    """
    assert reference.coordinates == fitted.coordinates
    reference_values = reference.gev_value_name_to_serie
    fitted_values = fitted.gev_value_name_to_serie
    gev_param_name_to_error_serie = {}
    for gev_value_name in GevParams.GEV_VALUE_NAMES:
        error = 100 * relative_abs_error(reference_values[gev_value_name], fitted_values[gev_value_name])
        gev_param_name_to_error_serie[gev_value_name] = error
    return gev_param_name_to_error_serie
