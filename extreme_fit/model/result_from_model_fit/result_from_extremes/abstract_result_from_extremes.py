import io
from collections import OrderedDict
from contextlib import redirect_stdout

import numpy as np
import pandas as pd
import rpy2
from rpy2 import robjects

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.utils import r


class AbstractResultFromExtremes(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None, dim_to_coordinate=None) -> None:
        super().__init__(result_from_fit)
        self.param_name_to_dim = param_name_to_dim
        self.dim_to_coordinate = dim_to_coordinate

    @property
    def summary_name_to_value(self):
        # Warning print will not work in this part
        f = io.StringIO()
        with redirect_stdout(f):
            summary = r('summary.fevd.mle_fixed')(self.result_from_fit)
            return self.get_python_dictionary(summary)

    @property
    def results(self):
        return self.get_python_dictionary(self.name_to_value['results'])

    @property
    def bic(self):
        return np.array(self.summary_name_to_value['BIC'])[0]

    @property
    def aic(self):
        return np.array(self.summary_name_to_value['AIC'])[0]

    @property
    def nllh(self):
        return np.array(self.results['value'])[0]

    @property
    def is_non_stationary(self):
        return len(self.param_name_to_dim) > 0

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
        if self.param_name_to_dim:
            print("here")
            if isinstance(transformed_temporal_covariate, (int, float, np.int, np.float, np.int64)):
                print("here2")
                d = {GevParams.greek_letter_from_param_name_confidence_interval(param_name) + '1': r.c(transformed_temporal_covariate) for
                     param_name in self.param_name_to_dim.keys()}
            elif isinstance(transformed_temporal_covariate, np.ndarray):
                d = OrderedDict()
                linearity_in_shape = GevParams.SHAPE in self.param_name_to_dim
                nb_calls = 2  # or 4 (1 and 3 did not work for the test)
                for param_name in GevParams.PARAM_NAMES:
                    suffix = '0' if param_name in self.param_name_to_dim else ''
                    covariate = np.array([1] * nb_calls)
                    d2 = {GevParams.greek_letter_from_param_name_confidence_interval(param_name, linearity_in_shape) + suffix: r.c(covariate)}
                    d.update(d2)
                    if param_name in self.param_name_to_dim:
                        for coordinate_idx, _ in self.param_name_to_dim[param_name]:
                            idx_str = str(coordinate_idx + 1)
                            covariate = float(transformed_temporal_covariate.copy()[coordinate_idx])
                            covariate = np.array([covariate] * nb_calls)
                            d2 = {GevParams.greek_letter_from_param_name_confidence_interval(param_name, linearity_in_shape) + idx_str: r.c(covariate)}
                            d.update(d2)

            else:
                raise NotImplementedError

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
