from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from experiment.trend_analysis.univariate_test.gev_trend_test_two_parameters import GevLocationAndScaleTrendTest
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import \
    NonStationaryLocationAndScaleModel, NonStationaryLocationStationModel, NonStationaryScaleStationModel
import numpy as np


class AbstractComparisonNonStationaryModel(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        raise NotImplementedError


class AbstractComparisonNonStationaryModelOneParameter(AbstractComparisonNonStationaryModel):

    @property
    def degree_freedom_chi2(self) -> int:
        return 1

    @property
    def test_sign(self) -> int:
        # Test sign correspond to the difference between the 2 likelihoods
        # Therefore, colors sum up which non stationary model explain best the data
        return np.sign(self.likelihood_ratio)


class ComparisonAgainstMu(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year, stationary_model_class=NonStationaryLocationStationModel)


class ComparisonAgainstSigma(AbstractComparisonNonStationaryModelOneParameter, GevLocationAndScaleTrendTest):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year, stationary_model_class=NonStationaryScaleStationModel)
