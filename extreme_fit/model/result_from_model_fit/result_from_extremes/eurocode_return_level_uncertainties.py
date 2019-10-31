from enum import Enum

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    ExtractEurocodeReturnLevelFromMyBayesianExtremes, ExtractEurocodeReturnLevelFromCiMethod
from experiment.trend_analysis.univariate_test.utils import load_temporal_coordinates_and_dataset, \
    fitted_linear_margin_estimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes


def compute_eurocode_confidence_interval(last_year_for_the_data, smooth_maxima_x_y, model_class, ci_method):
    years, smooth_maxima = smooth_maxima_x_y
    idx = years.index(last_year_for_the_data) + 1
    years, smooth_maxima = years[:idx], smooth_maxima[:idx]
    return EurocodeConfidenceIntervalFromExtremes.from_maxima_years_model_class(smooth_maxima, years, model_class, ci_method)


class EurocodeConfidenceIntervalFromExtremes(object):
    YEAR_OF_INTEREST = 2017

    def __init__(self, posterior_mean, poster_uncertainty_interval):
        self.posterior_mean = posterior_mean
        self.poster_uncertainty_interval = poster_uncertainty_interval

    @classmethod
    def from_estimator_extremes(cls, estimator_extremes: LinearMarginEstimator,
                                ci_method: ConfidenceIntervalMethodFromExtremes):
        if ci_method == ConfidenceIntervalMethodFromExtremes.my_bayes:
            extractor = ExtractEurocodeReturnLevelFromMyBayesianExtremes(estimator_extremes, ci_method, cls.YEAR_OF_INTEREST)
        else:
            extractor = ExtractEurocodeReturnLevelFromCiMethod(estimator_extremes, ci_method, cls.YEAR_OF_INTEREST)
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



