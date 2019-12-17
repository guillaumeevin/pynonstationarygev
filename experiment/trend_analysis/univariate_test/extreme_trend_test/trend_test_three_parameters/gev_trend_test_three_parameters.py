from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import \
    GevLocationAndScaleTrendTest
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.extreme_trend_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel, StationaryTemporalModel, NonStationaryLocationAndScaleGumbelModel, \
    GumbelTemporalModel
from extreme_fit.distribution.gev.gev_params import GevParams
from root_utils import classproperty


class GevTrendTestThreeParameters(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        return 3


class GevLocationAndScaleTrendTestAgainstGumbel(GevTrendTestThreeParameters):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationAndScaleTemporalModel,
                         constrained_model_class=GumbelTemporalModel,
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

    @classproperty
    def label(self):
        return super().label % '\\zeta_0, \\mu_1, \\sigma_1'

    @classproperty
    def marker(self):
        return 'D'

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
        return 5