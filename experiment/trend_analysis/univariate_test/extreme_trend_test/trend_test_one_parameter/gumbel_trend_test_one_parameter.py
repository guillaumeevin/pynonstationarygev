# Comparison with the Gumbel model
from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevTrendTestOneParameter, GevTrendTestOneParameterAgainstStationary
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    GumbelTemporalModel, NonStationaryLocationGumbelModel, NonStationaryScaleGumbelModel
from root_utils import classproperty


class GumbelVersusGumbel(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=GumbelTemporalModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level)

    @property
    def is_significant(self) -> bool:
        return False

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
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

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=StationaryTemporalModel,
                         constrained_model_class=GumbelTemporalModel,
                         quantile_level=quantile_level)

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
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

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationGumbelModel,
                         gev_param_name=GevParams.LOC,
                         quantile_level=quantile_level,
                         constrained_model_class=GumbelTemporalModel)

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
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

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryScaleGumbelModel,
                         gev_param_name=GevParams.SCALE,
                         quantile_level=quantile_level,
                         constrained_model_class=GumbelTemporalModel)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params.time_derivative_of_return_level(
            p=self.quantile_level,
            sigma1=self.non_stationary_linear_coef)

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
        return 3

    @classproperty
    def label(self):
        return super().label % '\\sigma_1'

    @classproperty
    def marker(self):
        return 11
