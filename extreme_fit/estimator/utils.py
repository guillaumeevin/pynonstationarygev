import numpy as np

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.margin_model.margin_function.linear_margin_function import LinearMarginFunction


def load_margin_function(estimator: AbstractEstimator, margin_model: LinearMarginModel,
                         margin_function_class=LinearMarginFunction, coef_dict=None):
    if coef_dict is None:
        coef_dict = estimator.result_from_model_fit.margin_coef_ordered_dict
    return margin_function_class.from_coef_dict(coordinates=estimator.dataset.coordinates,
                                                gev_param_name_to_dims=margin_model.margin_function_start_fit.gev_param_name_to_dims,
                                                coef_dict=coef_dict,
                                                starting_point=margin_model.starting_point)


def compute_nllh(estimator: AbstractEstimator, maxima, coordinate_temp, margin_model: LinearMarginModel,
                 margin_function_class=LinearMarginFunction, coef_dict=None):
    margin_function = load_margin_function(estimator, margin_model, margin_function_class, coef_dict)
    nllh = 0
    for maximum, year in zip(maxima[0], coordinate_temp.values):
        gev_params = margin_function.get_gev_params(year, is_transformed=False)
        p = gev_params.density(maximum)
        nllh -= np.log(p)
    return nllh
