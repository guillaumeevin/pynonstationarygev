from typing import List


import numpy as np
from cached_property import cached_property

from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.utils import load_temporal_coordinates_and_dataset, \
    fitted_linear_margin_estimator
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_extremes import ResultFromExtremes


class ExtractEurocodeReturnLevelFromExtremes(object):

    def __init__(self, estimator: LinearMarginEstimator, year_of_interest: int = 2017):
        self.estimator = estimator
        self.result_from_fit = self.estimator.result_from_model_fit # type: ResultFromExtremes
        self.year_of_interest = year_of_interest

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
        return [m.get_gev_params(coordinate=np.array([self.year_of_interest]), is_transformed=False)
                for m in self.margin_functions_from_fit]

    @cached_property
    def posterior_eurocode_return_level_samples_for_year_of_interest(self) -> np.ndarray:
        """We divide by 100 to transform the snow water equivalent into snow load"""
        return np.array([p.quantile(EUROCODE_QUANTILE) for p in self.gev_params_from_fit_for_year_of_interest]) / 100

    @property
    def posterior_mean_eurocode_return_level_for_the_year_of_interest(self) -> np.ndarray:
        return np.mean(self.posterior_eurocode_return_level_samples_for_year_of_interest)

    @property
    def posterior_eurocode_return_level_uncertainty_interval_for_the_year_of_interest(self):
        bottom_and_upper_quantile = (0.05, 0.95)
        return [np.quantile(self.posterior_eurocode_return_level_samples_for_year_of_interest, q=q)
                for q in bottom_and_upper_quantile]


class EurocodeLevelUncertaintyFromExtremes(object):

    YEAR_OF_INTEREST = 2017

    def __init__(self, posterior_mean, poster_uncertainty_interval):
        self.posterior_mean = posterior_mean
        self.poster_uncertainty_interval = poster_uncertainty_interval

    @classmethod
    def from_estimator_extremes(cls, estimator_extremes: LinearMarginEstimator):
        extractor = ExtractEurocodeReturnLevelFromExtremes(estimator_extremes, cls.YEAR_OF_INTEREST)
        return cls(extractor.posterior_mean_eurocode_return_level_for_the_year_of_interest,
                   extractor.posterior_eurocode_return_level_uncertainty_interval_for_the_year_of_interest)

    @classmethod
    def from_maxima_years_model_class(cls, maxima, years, model_class):
        # Load coordinates and dataset
        coordinates, dataset = load_temporal_coordinates_and_dataset(maxima, years)
        # Fitted estimator
        fitted_estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year=1958,
                                                          fit_method=AbstractTemporalLinearMarginModel.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR)
        # Load object from result from extremes
        return cls.from_estimator_extremes(fitted_estimator)