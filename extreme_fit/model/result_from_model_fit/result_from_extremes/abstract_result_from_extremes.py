import numpy as np
import pandas as pd
import rpy2
from rpy2 import robjects

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.utils import r


class AbstractResultFromExtremes(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None) -> None:
        super().__init__(result_from_fit)
        self.gev_param_name_to_dim = gev_param_name_to_dim

    @property
    def results(self):
        return self.get_python_dictionary(self.name_to_value['results'])

    @property
    def nllh(self):
        return np.array(self.results['value'])[0]

    @property
    def is_non_stationary(self):
        return len(self.gev_param_name_to_dim) > 0

    def load_dataframe_from_r_matrix(self, name):
        r_matrix = self.name_to_value[name]
        return pd.DataFrame(np.array(r_matrix), columns=r.colnames(r_matrix))

    def confidence_interval_method(self, quantile_level, alpha_interval, transformed_temporal_covariate, ci_method):
        return_period = round(1 / (1 - quantile_level))
        common_kwargs = {
            'return.period': return_period,
            'alpha': alpha_interval,
            'tscale': False,
            'type': r.c("return.level")
        }
        if self.gev_param_name_to_dim:
            d = {GevParams.greek_letter_from_gev_param_name(gev_param_name) + '1': r.c(transformed_temporal_covariate) for
                 gev_param_name in self.gev_param_name_to_dim.keys()}
            kwargs = {
                "vals": r.list(**d
                               )
            }
            qcov = r("make.qcov")(self.result_from_fit,
                                  **kwargs)
            common_kwargs['qcov'] = qcov
        try:
            mean_estimate, confidence_interval = self._confidence_interval_method(common_kwargs, ci_method, return_period)
        except rpy2.rinterface.RRuntimeError:
            mean_estimate, confidence_interval = np.nan, (np.nan, np.nan)
        return mean_estimate, confidence_interval

    def _confidence_interval_method(self, common_kwargs, ci_method, return_period):
        raise NotImplementedError
