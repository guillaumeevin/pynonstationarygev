import numpy as np
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ci_method_to_method_name
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r


class ResultFromMleExtremes(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None,
                 dim_to_coordinate=None,
                 type_for_mle="GEV",
                 param_name_to_name_of_the_climatic_effects=None,
                 param_name_to_climate_coordinates_with_effects=None) -> None:
        super().__init__(result_from_fit, param_name_to_dim, dim_to_coordinate)
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
        self.param_name_to_name_of_the_climatic_effects = param_name_to_name_of_the_climatic_effects
        self.type_for_mle = type_for_mle

    @property
    def param_name_to_name_of_the_climatic_effects_to_load_margin_function(self):
        return self.param_name_to_name_of_the_climatic_effects

    @property
    def param_name_to_climate_coordinates_with_effects_to_load_margin_function(self):
        return self.param_name_to_climate_coordinates_with_effects

    @property
    def margin_coef_ordered_dict(self):
        values = self.name_to_value['results']
        d = self.get_python_dictionary(values)
        if 'par' in d:
            values = {i: param for i, param in enumerate(np.array(d['par']))}
        else:
            values = {i: np.array(v)[0] for i, v in enumerate(d.values())}
        return get_margin_coef_ordered_dict(self.param_name_to_dims, values, self.type_for_mle,
                                            dim_to_coordinate_name=self.dim_to_coordinate,
                                            param_name_to_name_of_the_climatic_effects=self.param_name_to_name_of_the_climatic_effects)

    def _confidence_interval_method(self, common_kwargs, ci_method, return_period):
        method_name = ci_method_to_method_name[ci_method]
        mle_ci_parameters = {
            'method': method_name,
            # xrange = NULL, nint = 20
        }
        res = r('ci.fevd.mle_fixed')(self.result_from_fit, **mle_ci_parameters, **common_kwargs)
        if self.is_non_stationary:
            b = np.array(res)
            a = b[0]
            lower, mean_estimate, upper, _ = a
        else:
            d = self.get_python_dictionary(res)
            keys = ['{}-year return level'.format(return_period), '95% lower CI', '95% upper CI']
            mean_estimate, lower, upper = [np.array(d[k])[0] for k in keys]

        return mean_estimate, (lower, upper)


