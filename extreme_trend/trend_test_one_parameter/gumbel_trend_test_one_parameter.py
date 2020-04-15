# Comparison with the Gumbel model
from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_trend.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevTrendTestOneParameter, GevTrendTestOneParameterAgainstStationary
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    GumbelTemporalModel, NonStationaryLocationGumbelModel, NonStationaryScaleGumbelModel
from root_utils import classproperty


class GumbelVersusGumbel(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=GumbelTemporalModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @property
    def is_significant(self) -> bool:
        return False

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 2

    @classproperty
    def label(self):
        return super().label % '0'

    @classproperty
    def marker(self):
        return 'x'

    def _slope_strength(self):
        return 0.0


class GevStationaryVersusGumbel(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=StationaryTemporalModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 3

    def _slope_strength(self):
        return 0.0

    @classproperty
    def label(self):
        return super().label % '\\zeta_0'

    @classproperty
    def marker(self):
        return 'X'


class GumbelLocationTrendTest(GevTrendTestOneParameterAgainstStationary):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationGumbelModel,
                         gev_param_name=GevParams.LOC,
                         quantile_level=quantile_level,
                         constrained_model_class=GumbelTemporalModel, fit_method=fit_method)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 3

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params.time_derivative_of_return_level(p=self.quantile_level,
                                                                                       mu1=self.non_stationary_linear_coef)

    @classproperty
    def label(self):
        return super().label % '\\mu_1'

    @classproperty
    def marker(self):
        return '.'


class GumbelScaleTrendTest(GevTrendTestOneParameterAgainstStationary):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryScaleGumbelModel,
                         gev_param_name=GevParams.SCALE,
                         quantile_level=quantile_level,
                         constrained_model_class=GumbelTemporalModel,
                         fit_method=fit_method)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params.time_derivative_of_return_level(
            p=self.quantile_level,
            sigma1=self.non_stationary_linear_coef)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 3

    @classproperty
    def label(self):
        return super().label % '\\sigma_1'

    @classproperty
    def marker(self):
        return 11
