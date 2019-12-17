from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.extreme_trend_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel, StationaryTemporalModel, NonStationaryLocationAndScaleGumbelModel, \
    GumbelTemporalModel
from extreme_fit.distribution.gev.gev_params import GevParams


class GevTrendTestTwoParameters(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        return 2


class GevLocationAndScaleTrendTest(GevTrendTestTwoParameters):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationAndScaleTemporalModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level)

    @property
    def mu1(self):
        return self.get_non_stationary_linear_coef(gev_param_name=GevParams.LOC)

    @property
    def sigma1(self):
        return self.get_non_stationary_linear_coef(gev_param_name=GevParams.SCALE)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params.time_derivative_of_return_level(p=self.quantile_level,
                                                                                       mu1=self.mu1,
                                                                                       sigma1=self.sigma1)

    @property
    def mean_difference_same_sign_as_slope_strenght(self) -> bool:
        zeta0 = self.unconstrained_estimator_gev_params.shape
        mean_difference = self.mean_difference(zeta0=zeta0, mu1=self.mu1, sigma1=self.sigma1)
        return self.same_sign(mean_difference, self._slope_strength())

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return self.same_sign(self.sigma1, self._slope_strength())

