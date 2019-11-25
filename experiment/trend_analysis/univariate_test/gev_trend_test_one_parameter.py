from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel
from extreme_fit.distribution.gev.gev_params import GevParams


class GevTrendTestOneParameter(AbstractGevTrendTest):

    def __init__(self, years, maxima, starting_year, unconstrained_model_class, gev_param_name, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=unconstrained_model_class,
                         quantile_level=quantile_level)
        self.gev_param_name = gev_param_name

    @property
    def degree_freedom_chi2(self) -> int:
        return 1

    @property
    def non_stationary_linear_coef(self):
        return self.get_non_stationary_linear_coef(gev_param_name=self.gev_param_name)


class GevLocationTrendTest(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationTemporalModel,
                         gev_param_name=GevParams.LOC,
                         quantile_level=quantile_level)

    def _slope_strength(self):
        return self.non_stationary_constant_gev_params.time_derivative_of_return_level(p=self.quantile_level,
                                                                                       mu1=self.non_stationary_linear_coef)

    @property
    def mean_difference_same_sign_as_slope_strenght(self) -> bool:
        zeta0 = self.non_stationary_constant_gev_params.shape
        mean_difference = self.mean_difference(zeta0=zeta0, mu1=self.non_stationary_linear_coef)
        return self.same_sign(mean_difference, self._slope_strength())

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return False


class GevScaleTrendTest(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryScaleTemporalModel,
                         gev_param_name=GevParams.SCALE,
                         quantile_level=quantile_level)

    def _slope_strength(self):
        return self.non_stationary_constant_gev_params.time_derivative_of_return_level(
            p=self.quantile_level,
            sigma1=self.non_stationary_linear_coef)

    @property
    def mean_difference_same_sign_as_slope_strenght(self) -> bool:
        zeta0 = self.non_stationary_constant_gev_params.shape
        mean_difference = self.mean_difference(zeta0=zeta0, sigma1=self.non_stationary_linear_coef)
        return self.same_sign(mean_difference, self._slope_strength())

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        sigma1 = self.non_stationary_linear_coef
        return self.same_sign(sigma1, self._slope_strength())


class GevShapeTrendTest(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryShapeTemporalModel,
                         gev_param_name=GevParams.SHAPE,
                         quantile_level=quantile_level)
