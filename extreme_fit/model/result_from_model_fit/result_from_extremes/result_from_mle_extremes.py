import numpy as np
import rpy2

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ci_method_to_method_name
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r


class ResultFromMleExtremes(AbstractResultFromExtremes):

    @property
    def margin_coef_ordered_dict(self):
        values = self.name_to_value['results']
        d = self.get_python_dictionary(values)
        values = {i: param for i, param in enumerate(np.array(d['par']))}
        return get_margin_coef_ordered_dict(self.gev_param_name_to_dim, values)

    def _confidence_interval_method(self, common_kwargs, ci_method):
        method_name = ci_method_to_method_name[ci_method]
        mle_ci_parameters = {
                'method': method_name,
            # xrange = NULL, nint = 20
        }
        try:
            res = r.ci(self.result_from_fit, **mle_ci_parameters, **common_kwargs)
        except rpy2.rinterface.RRuntimeError:
            return np.nan, (np.nan, np.nan)
        if self.is_non_stationary:
            a = np.array(res)[0]
            lower, mean_estimate, upper, _ = a
        else:
            d = self.get_python_dictionary(res)
            keys = ['50-year return level', '95% lower CI', '95% upper CI']
            mean_estimate, lower, upper = [np.array(d[k])[0] for k in keys]
        return mean_estimate, (lower, upper)


