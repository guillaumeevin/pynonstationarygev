import numpy as np
import pandas as pd
from rpy2 import robjects

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import \
    AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_dict
from extreme_fit.model.utils import r


class ResultFromExtremes(AbstractResultFromModelFit):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None,
                 burn_in_percentage=0.1) -> None:
        super().__init__(result_from_fit)
        self.burn_in_percentage = burn_in_percentage
        self.gev_param_name_to_dim = gev_param_name_to_dim

    def load_dataframe_from_r_matrix(self, k):
        r_matrix = self.name_to_value[k]
        return pd.DataFrame(np.array(r_matrix), columns=r.colnames(r_matrix))

    @property
    def chain_info(self):
        return self.load_dataframe_from_r_matrix('chain.info')

    @property
    def results(self):
        return self.load_dataframe_from_r_matrix('results')

    @property
    def df_posterior_samples(self) -> pd.DataFrame:
        df_all_samples = pd.concat([self.results.iloc[:, :-1], self.chain_info.iloc[:, -2:]], axis=1)
        # Remove the burn in period
        burn_in_last_index = int(self.burn_in_percentage * len(df_all_samples))
        df_posterior_samples = df_all_samples.iloc[burn_in_last_index:, :]
        return df_posterior_samples

    def get_coef_dict_from_posterior_sample(self, s: pd.Series):
        assert len(s) >= 3
        values = {i: v for i, v in enumerate(s)}
        return get_margin_coef_dict(self.gev_param_name_to_dim, values)

    @property
    def margin_coef_dict(self):
        """ It is the coef for the margin function corresponding to the mean posterior parameters """
        mean_posterior_parameters = self.df_posterior_samples.iloc[:, :-2].mean(axis=0)
        return self.get_coef_dict_from_posterior_sample(mean_posterior_parameters)



# @property
    # def

    # @property
    # def margin_coef_dict(self):
    #     assert self.gev_param_name_to_dim is not None
    #     # Build the Coeff dict from gev_param_name_to_dim
    #     coef_dict = {}
    #     i = 0
    #     mle_values = self.name_to_value['mle']
    #     for gev_param_name in GevParams.PARAM_NAMES:
    #         # Add intercept
    #         intercept_coef_name = LinearCoef.coef_template_str(gev_param_name, LinearCoef.INTERCEPT_NAME).format(1)
    #         coef_dict[intercept_coef_name] = mle_values[i]
    #         i += 1
    #         # Add a potential linear temporal trend
    #         if gev_param_name in self.gev_param_name_to_dim:
    #             temporal_coef_name = LinearCoef.coef_template_str(gev_param_name,
    #                                                               AbstractCoordinates.COORDINATE_T).format(1)
    #             coef_dict[temporal_coef_name] = mle_values[i]
    #             i += 1
    #     return coef_dict

    # @property
    # def all_parameters(self):
    #     return self.margin_coef_dict
    #
    # @property
    # def nllh(self):
    #     return convertFloatVector_to_float(self.name_to_value['nllh'])
    #
    # @property
    # def deviance(self):
    #     return - 2 * self.nllh
    #
    # @property
    # def convergence(self) -> str:
    #     return convertFloatVector_to_float(self.name_to_value['conv']) == 0
