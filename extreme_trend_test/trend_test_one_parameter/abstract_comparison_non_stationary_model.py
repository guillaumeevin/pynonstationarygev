from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevTrendTestOneParameter
from extreme_trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel
import numpy as np


class AbstractComparisonNonStationaryModelOneParameter(GevTrendTestOneParameter):

    @property
    def test_sign(self) -> int:
        # Test sign correspond to the difference between the 2 likelihoods
        # Therefore, colors sum up which non stationary model explain best the data
        return np.sign(self.likelihood_ratio)


class ComparisonAgainstMu(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE, fit_method=TemporalMarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, constrained_model_class=NonStationaryLocationTemporalModel,
                         quantile_level=quantile_level, fit_method=fit_method)


class ComparisonAgainstSigma(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE, fit_method=TemporalMarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, constrained_model_class=NonStationaryScaleTemporalModel,
                         quantile_level=quantile_level, fit_method=fit_method)
