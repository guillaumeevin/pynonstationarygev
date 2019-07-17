from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import \
    NonStationaryLocationAndScaleModel
from extreme_estimator.margin_fits.gev.gev_params import GevParams


class GevTrendTestTwoParameters(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        return 2


class GevLocationAndScaleTrendTest(GevTrendTestTwoParameters):

    def __init__(self, years, maxima, starting_year):
        super().__init__(years, maxima, starting_year,
                         NonStationaryLocationAndScaleModel)

    def _slope_strength(self):
        mu1 = self.get_non_stationary_linear_coef(gev_param_name=GevParams.LOC)
        sigma1 = self.get_non_stationary_linear_coef(gev_param_name=GevParams.SCALE)
        return self.non_stationary_constant_gev_params.quantile_strength_evolution_ratio(p=self.quantile_for_strength,
                                                                                         mu1=mu1,
                                                                                         sigma1=sigma1)
