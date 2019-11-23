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


def compute_eurocode_confidence_interval(smooth_maxima_x_y, model_class, ci_method, temporal_covariate):
    years, smooth_maxima = smooth_maxima_x_y
    return EurocodeConfidenceIntervalFromExtremes.from_maxima_years_model_class(smooth_maxima, years, model_class, temporal_covariate, ci_method)


class EurocodeConfidenceIntervalFromExtremes(object):

    def __init__(self, mean_estimate, confidence_interval):
        self.mean_estimate = mean_estimate
        self.confidence_interval = confidence_interval

    @property
    def triplet(self):
        return self.confidence_interval[0], self.mean_estimate, self.confidence_interval[1]

    @classmethod
    def from_estimator_extremes(cls, estimator_extremes: LinearMarginEstimator,
                                ci_method: ConfidenceIntervalMethodFromExtremes,
                                temporal_covariate: int):
        if ci_method == ConfidenceIntervalMethodFromExtremes.my_bayes:
            extractor = ExtractEurocodeReturnLevelFromMyBayesianExtremes(estimator_extremes, ci_method, temporal_covariate)
        else:
            extractor = ExtractEurocodeReturnLevelFromCiMethod(estimator_extremes, ci_method, temporal_covariate)
        return cls(extractor.mean_estimate,  extractor.confidence_interval)

    @classmethod
    def from_maxima_years_model_class(cls, maxima, years, model_class,
                                      temporal_covariate,
                                      ci_method=ConfidenceIntervalMethodFromExtremes.ci_bayes):
        # Load coordinates and dataset
        coordinates, dataset = load_temporal_coordinates_and_dataset(maxima, years)
        # Select fit method depending on the ci_method
        if ci_method in [ConfidenceIntervalMethodFromExtremes.ci_bayes,
                         ConfidenceIntervalMethodFromExtremes.my_bayes]:
            fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        else:
            fit_method = TemporalMarginFitMethod.extremes_fevd_mle
        # Fitted estimator
        fitted_estimator = fitted_linear_margin_estimator(model_class, coordinates, dataset, starting_year=1958,
                                                          fit_method=fit_method)
        # Load object from result from extremes
        return cls.from_estimator_extremes(fitted_estimator, ci_method, temporal_covariate)



