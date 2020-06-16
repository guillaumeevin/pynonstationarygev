from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_fit.model.margin_model.polynomial_margin_model import NonStationaryQuadraticLocationModel, \
    NonStationaryQuadraticLocationGumbelModel, NonStationaryQuadraticScaleGumbelModel
from extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters import \
    GevTrendTestTwoParameters
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleGumbelModel, GumbelTemporalModel
from root_utils import classproperty


class GumbelLocationAndScaleTrendTest(GevTrendTestTwoParameters):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationAndScaleGumbelModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4

    @property
    def mu1(self):
        return self.get_non_stationary_linear_coef(param_name=GevParams.LOC)

    @property
    def sigma1(self):
        return self.get_non_stationary_linear_coef(param_name=GevParams.SCALE)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params_last_year.time_derivative_of_return_level(p=self.quantile_level,
                                                                                                 mu1=self.mu1,
                                                                                                 sigma1=self.sigma1)
    @classproperty
    def label(self):
        return super().label % '\\mu_1, \\sigma_1'

    @classproperty
    def marker(self):
        return 'd'

class GumbelLocationQuadraticTrendTest(GevTrendTestTwoParameters):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryQuadraticLocationGumbelModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4

class GumbelScaleQuadraticTrendTest(GevTrendTestTwoParameters):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryQuadraticScaleGumbelModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4

