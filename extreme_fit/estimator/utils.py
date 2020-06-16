import numpy as np

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction


def load_margin_function(estimator: AbstractEstimator, margin_model: LinearMarginModel, coef_dict=None):
    if coef_dict is None:
        coef_dict = estimator.result_from_model_fit.margin_coef_ordered_dict
    margin_function_class = type(margin_model.margin_function)
    return margin_function_class.from_coef_dict(coordinates=estimator.dataset.coordinates,
                                                param_name_to_dims=margin_model.param_name_to_list_for_result,
                                                coef_dict=coef_dict,
                                                starting_point=margin_model.starting_point)



