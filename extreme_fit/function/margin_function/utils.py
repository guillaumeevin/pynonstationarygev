from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_fit.distribution.gev.gev_params import GevParams


def relative_abs_error(reference_value, fitted_value):
    # todo: handle the case when the location, or the shape we aim to estimate might be equal to zero
    return (reference_value - fitted_value).abs() / reference_value


def error_dict_between_margin_functions(reference: AbstractMarginFunction, fitted: AbstractMarginFunction):
    """
    Return a serie, indexed by the same index as the coordinates
    Each value correspond to the error for this coordinate
    :param reference:
    :param fitted:
    :return:
    """
    assert reference.coordinates == fitted.coordinates, \
        'Coordinates have either been resampled or the split is not the same'
    reference_values = reference.gev_value_name_to_serie
    fitted_values = fitted.gev_value_name_to_serie
    gev_param_name_to_error_serie = {}
    for gev_value_name in GevParams.SUMMARY_NAMES:
        error = 100 * relative_abs_error(reference_values[gev_value_name], fitted_values[gev_value_name])
        gev_param_name_to_error_serie[gev_value_name] = error
    return gev_param_name_to_error_serie
