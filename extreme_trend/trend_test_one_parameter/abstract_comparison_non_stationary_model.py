from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_trend.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevTrendTestOneParameter
from extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel
import numpy as np


class AbstractComparisonNonStationaryModelOneParameter(GevTrendTestOneParameter):
    pass


class ComparisonAgainstMu(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, constrained_model_class=NonStationaryLocationTemporalModel,
                         quantile_level=quantile_level, fit_method=fit_method)


class ComparisonAgainstSigma(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, constrained_model_class=NonStationaryScaleTemporalModel,
                         quantile_level=quantile_level, fit_method=fit_method)
