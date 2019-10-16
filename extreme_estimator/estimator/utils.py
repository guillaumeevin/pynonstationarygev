from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction


def load_margin_function(estimator: AbstractEstimator, margin_model: LinearMarginModel, margin_function_class=LinearMarginFunction):
    return margin_function_class.from_coef_dict(coordinates=estimator.dataset.coordinates,
                                                gev_param_name_to_dims=margin_model.margin_function_start_fit.gev_param_name_to_dims,
                                                coef_dict=estimator.result_from_model_fit.margin_coef_dict,
                                                starting_point=margin_model.starting_point)
