import numpy as np

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction


def load_margin_function(estimator: AbstractEstimator, margin_model: LinearMarginModel,
                         coef_dict=None, log_scale=None, param_name_to_name_of_the_climatic_effects=None,
                         param_name_to_climate_coordinates_with_effects=None):
    if coef_dict is None:
        coef_dict = estimator.result_from_model_fit.margin_coef_ordered_dict
    if log_scale is None:
        log_scale = estimator.result_from_model_fit.log_scale
    if param_name_to_name_of_the_climatic_effects is None:
        param_name_to_name_of_the_climatic_effects = estimator.result_from_model_fit.param_name_to_name_of_the_climatic_effects_to_load_margin_function
    if param_name_to_climate_coordinates_with_effects is None:
        param_name_to_climate_coordinates_with_effects = estimator.result_from_model_fit.param_name_to_climate_coordinates_with_effects_to_load_margin_function

    margin_function_class = type(margin_model.margin_function)
    return margin_function_class.from_coef_dict(coordinates=estimator.dataset.coordinates,
                                                param_name_to_dims=margin_model.param_name_to_list_for_result,
                                                coef_dict=coef_dict,
                                                starting_point=margin_model.starting_point,
                                                log_scale=log_scale,
                                                param_name_to_name_of_the_climatic_effects=param_name_to_name_of_the_climatic_effects,
                                                param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects)



