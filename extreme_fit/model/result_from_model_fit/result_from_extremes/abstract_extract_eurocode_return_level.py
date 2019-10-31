from typing import List

import numpy as np
from cached_property import cached_property

from experiment.eurocode_data.utils import EUROCODE_QUANTILE, YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    ResultFromBayesianExtremes


class AbstractExtractEurocodeReturnLevel(object):
    ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.05

    def __init__(self, estimator: LinearMarginEstimator,
                 ci_method,
                 year_of_interest: int = YEAR_OF_INTEREST_FOR_RETURN_LEVEL,
                 ):
        self.ci_method = ci_method
        self.estimator = estimator
        self.result_from_fit = self.estimator.result_from_model_fit

        self.year_of_interest = year_of_interest

        self.eurocode_quantile = EUROCODE_QUANTILE
        self.alpha_for_confidence_interval = self.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY


class ExtractEurocodeReturnLevelFromCiMethod(AbstractExtractEurocodeReturnLevel):
    pass


class ExtractEurocodeReturnLevelFromMyBayesianExtremes(AbstractExtractEurocodeReturnLevel):
    result_from_fit: ResultFromBayesianExtremes

    def __init__(self, estimator: LinearMarginEstimator, ci_method,
                 year_of_interest: int = YEAR_OF_INTEREST_FOR_RETURN_LEVEL):
        super().__init__(estimator, ci_method, year_of_interest)
        assert isinstance(self.result_from_fit, ResultFromBayesianExtremes)

    @property
    def margin_functions_from_fit(self) -> List[LinearMarginFunction]:
        margin_functions = []
        for _, s in self.result_from_fit.df_posterior_samples.iterrows():
            coef_dict = self.result_from_fit.get_coef_dict_from_posterior_sample(s)
            margin_function = load_margin_function(self.estimator, self.estimator.margin_model, coef_dict=coef_dict)
            margin_functions.append(margin_function)
        return margin_functions

    @property
    def gev_params_from_fit_for_year_of_interest(self) -> List[GevParams]:
        return [margin_function.get_gev_params(coordinate=np.array([self.year_of_interest]), is_transformed=False)
                for margin_function in self.margin_functions_from_fit]

    @cached_property
    def posterior_eurocode_return_level_samples_for_year_of_interest(self) -> np.ndarray:
        """We divide by 100 to transform the snow water equivalent into snow load"""
        return np.array(
            [p.quantile(self.eurocode_quantile) for p in self.gev_params_from_fit_for_year_of_interest]) / 100

    @property
    def posterior_mean_eurocode_return_level_for_the_year_of_interest(self) -> np.ndarray:
        return np.mean(self.posterior_eurocode_return_level_samples_for_year_of_interest)

    @property
    def posterior_eurocode_return_level_uncertainty_interval_for_the_year_of_interest(self):
        # Bottom and upper quantile correspond to the quantile
        bottom_quantile = self.alpha_for_confidence_interval / 2
        bottom_and_upper_quantile = (bottom_quantile, 1 - bottom_quantile)
        return [np.quantile(self.posterior_eurocode_return_level_samples_for_year_of_interest, q=q)
                for q in bottom_and_upper_quantile]