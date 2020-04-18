import numpy as np
import pandas as pd
from cached_property import cached_property
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict
from extreme_fit.model.utils import r


class ResultFromBayesianExtremes(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, param_name_to_dim=None,
                 burn_in_percentage=0.5) -> None:
        super().__init__(result_from_fit, param_name_to_dim)
        self.burn_in_percentage = burn_in_percentage

    @property
    def burn_in_nb(self):
        return int(self.burn_in_percentage * len(self.df_all_samples))

    @property
    def chain_info(self):
        return self.load_dataframe_from_r_matrix('chain.info')

    @property
    def results(self):
        return self.load_dataframe_from_r_matrix('results')

    @cached_property
    def df_all_samples(self):
        return pd.concat([self.results.iloc[:, :-1], self.chain_info.iloc[:, -2:]], axis=1)

    @property
    def df_posterior_samples(self) -> pd.DataFrame:
        return self.df_all_samples.iloc[self.burn_in_nb:, :]

    @property
    def df_posterior_parameters(self) -> pd.DataFrame:
        return self.df_posterior_samples.iloc[:, :-2]

    @property
    def mean_posterior_parameters(self):
        return self.df_posterior_parameters.mean(axis=0)

    @property
    def variance_posterior_parameters(self):
        return self.df_posterior_parameters.mean(axis=0)

    def get_coef_dict_from_posterior_sample(self, s: pd.Series):
        assert len(s) >= 3
        values = {i: v for i, v in enumerate(s)}
        return get_margin_coef_ordered_dict(self.param_name_to_dim, values)

    @property
    def margin_coef_ordered_dict(self):
        """ It is the coef for the margin function corresponding to the mean posterior parameters """
        mean_posterior_parameters = self.df_posterior_samples.iloc[:, :-2].mean(axis=0)
        return self.get_coef_dict_from_posterior_sample(mean_posterior_parameters)

    def _confidence_interval_method(self, common_kwargs, ci_method, return_period):
        bayesian_ci_parameters = {
                'burn.in': self.burn_in_nb,
                'FUN': "mean",
        }
        res = r.ci(self.result_from_fit, **bayesian_ci_parameters, **common_kwargs)
        if self.param_name_to_dim:
            a = np.array(res)[0]
            lower, mean_estimate, upper = a
        else:
            d = self.get_python_dictionary(res)
            keys = ['Posterior Mean {}-year level'.format(return_period), '95% lower CI', '95% upper CI']
            mean_estimate, lower, upper = [np.array(d[k])[0] for k in keys]
        return mean_estimate, (lower, upper)


