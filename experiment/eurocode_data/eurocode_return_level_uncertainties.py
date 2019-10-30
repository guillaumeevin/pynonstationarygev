from enum import Enum
from typing import List

import numpy as np
from cached_property import cached_property

from experiment.eurocode_data.utils import EUROCODE_QUANTILE, YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from experiment.trend_analysis.univariate_test.utils import load_temporal_coordinates_and_dataset, \
    fitted_linear_margin_estimator
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel, TemporalMarginFitMethod
from extreme_fit.model.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_extremes import ResultFromExtremes, ResultFromBayesianExtremes


class ConfidenceIntervalMethodFromExtremes(Enum):
    # Confidence interval from the ci function
    bayes = 0
    normal = 1
    boot = 2
    proflik = 3
    # Confidence interval from my functions
    my_bayes = 4


def compute_eurocode_level_uncertainty(last_year_for_the_data, smooth_maxima_x_y, model_class, ci_method):
    years, smooth_maxima = smooth_maxima_x_y
    idx = years.index(last_year_for_the_data) + 1
    years, smooth_maxima = years[:idx], smooth_maxima[:idx]
    return EurocodeLevelUncertaintyFromExtremes.from_maxima_years_model_class(smooth_maxima, years, model_class, ci_method)


class EurocodeLevelUncertaintyFromExtremes(object):
    YEAR_OF_INTEREST = 2017

    def __init__(self, posterior_mean, poster_uncertainty_interval):
        self.posterior_mean = posterior_mean
        self.poster_uncertainty_interval = poster_uncertainty_interval

    @classmethod
    def from_estimator_extremes(cls, estimator_extremes: LinearMarginEstimator,
                                ci_method: ConfidenceIntervalMethodFromExtremes):
        extractor = ExtractEurocodeReturnLevelFromExtremes(estimator_extremes, ci_method, cls.YEAR_OF_INTEREST)
        return cls(extractor.posterior_mean_eurocode_return_level_for_the_year_of_interest,
                   extractor.posterior_eurocode_return_level_uncertainty_interval_for_the_year_of_interest)

    @classmethod
    def from_maxima_years_model_class(cls, maxima, years, model_class,
                                      ci_method=ConfidenceIntervalMethodFromExtremes.bayes):
        # Load coordinates and dataset
        coordinates, dataset = load_temporal_coordinates_and_dataset(maxima, years)
        # Select fit method depending on the ci_method
        if ci_method in [ConfidenceIntervalMethodFromExtremes.bayes,
                         ConfidenceIntervalMethodFromExtremes.my_bayes]:
            fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        else:
            fit_method = TemporalMarginFitMethod.extremes_fevd_mle
        # Fitted estimator
        fitted_estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year=1958,
                                                          fit_method=fit_method)
        # Load object from result from extremes
        return cls.from_estimator_extremes(fitted_estimator, ci_method)


class ExtractFromExtremes(object):
    pass


class ExtractEurocodeReturnLevelFromExtremes(object):
    ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.05

    def __init__(self, estimator: LinearMarginEstimator,
                 ci_method,
                 year_of_interest: int = YEAR_OF_INTEREST_FOR_RETURN_LEVEL,
                 alpha_for_confidence_interval: int = ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY,
                 ):
        self.estimator = estimator
        self.result_from_fit = self.estimator.result_from_model_fit  # type: ResultFromBayesianExtremes
        assert isinstance(self.result_from_fit, ResultFromBayesianExtremes)
        self.year_of_interest = year_of_interest
        self.alpha_for_confidence_interval = alpha_for_confidence_interval

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
        return np.array([p.quantile(EUROCODE_QUANTILE) for p in self.gev_params_from_fit_for_year_of_interest]) / 100

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
