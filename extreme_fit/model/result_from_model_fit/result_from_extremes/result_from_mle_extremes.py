import numpy as np
import rpy2
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ci_method_to_method_name
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r


class ResultFromMleExtremes(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None,
                 type_for_mle="GEV") -> None:
        super().__init__(result_from_fit, gev_param_name_to_dim)
        self.type_for_mle = type_for_mle

    @property
    def margin_coef_ordered_dict(self):
        values = self.name_to_value['results']
        d = self.get_python_dictionary(values)
        values = {i: param for i, param in enumerate(np.array(d['par']))}
        return get_margin_coef_ordered_dict(self.gev_param_name_to_dim, values, self.type_for_mle)

    def _confidence_interval_method(self, common_kwargs, ci_method, return_period):
        method_name = ci_method_to_method_name[ci_method]
        mle_ci_parameters = {
                'method': method_name,
            # xrange = NULL, nint = 20
        }
        res = r('ci.fevd.mle_fixed')(self.result_from_fit, **mle_ci_parameters, **common_kwargs)
        if self.is_non_stationary:
            a = np.array(res)[0]
            lower, mean_estimate, upper, _ = a
        else:
            d = self.get_python_dictionary(res)
            keys = ['{}-year return level'.format(return_period), '95% lower CI', '95% upper CI']
            mean_estimate, lower, upper = [np.array(d[k])[0] for k in keys]
        return mean_estimate, (lower, upper)


