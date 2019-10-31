import pandas as pd
from rpy2 import robjects

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_result_from_extremes import \
    AbstractResultFromExtremes
from extreme_fit.model.result_from_model_fit.utils import get_margin_coef_ordered_dict


class ResultFromBayesianExtremes(AbstractResultFromExtremes):

    def __init__(self, result_from_fit: robjects.ListVector, gev_param_name_to_dim=None,
                 burn_in_percentage=0.5) -> None:
        super().__init__(result_from_fit, gev_param_name_to_dim)
        self.burn_in_percentage = burn_in_percentage

    def burn_in_nb(self, df_all_samples):
        return int(self.burn_in_percentage * len(df_all_samples))

    @property
    def chain_info(self):
        return self.load_dataframe_from_r_matrix('chain.info')

    @property
    def results(self):
        return self.load_dataframe_from_r_matrix('results')

    @property
    def df_posterior_samples(self) -> pd.DataFrame:
        df_all_samples = pd.concat([self.results.iloc[:, :-1], self.chain_info.iloc[:, -2:]], axis=1)
        df_posterior_samples = df_all_samples.iloc[self.burn_in_nb(df_all_samples):, :]
        return df_posterior_samples

    def get_coef_dict_from_posterior_sample(self, s: pd.Series):
        assert len(s) >= 3
        values = {i: v for i, v in enumerate(s)}
        return get_margin_coef_ordered_dict(self.gev_param_name_to_dim, values)

    @property
    def margin_coef_ordered_dict(self):
        """ It is the coef for the margin function corresponding to the mean posterior parameters """
        mean_posterior_parameters = self.df_posterior_samples.iloc[:, :-2].mean(axis=0)
        return self.get_coef_dict_from_posterior_sample(mean_posterior_parameters)
