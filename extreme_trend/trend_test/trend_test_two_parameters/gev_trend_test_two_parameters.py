from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import NonStationaryQuadraticLocationModel, \
    NonStationaryQuadraticScaleModel
from extreme_trend.trend_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_trend.trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevLocationTrendTest, GevScaleTrendTest
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel, StationaryTemporalModel, GumbelTemporalModel, \
    NonStationaryScaleAndShapeTemporalModel, NonStationaryLocationAndShapeTemporalModel
from extreme_fit.distribution.gev.gev_params import GevParams
from root_utils import classproperty


class GevTrendTestTwoParameters(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        return 2


class GevTrendTestTwoParametersAgainstGev(GevTrendTestTwoParameters):

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 5


class GevLocationAndShapeTrendTest(GevTrendTestTwoParametersAgainstGev):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationAndShapeTemporalModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level,
                         fit_method=fit_method)


class GevScaleAndShapeTrendTest(GevTrendTestTwoParametersAgainstGev):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryScaleAndShapeTemporalModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level,
                         fit_method=fit_method)
        
class GevQuadraticLocationTrendTest(GevTrendTestTwoParametersAgainstGev):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryQuadraticLocationModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

class GevQuadraticScaleTrendTest(GevTrendTestTwoParametersAgainstGev):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryQuadraticScaleModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level,
                         fit_method=fit_method)


class GevLocationAndScaleTrendTest(GevTrendTestTwoParametersAgainstGev):

    def __init__(self, years, maxima, starting_year, constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE, fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationAndScaleTemporalModel,
                         constrained_model_class=constrained_model_class,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

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
    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return self.same_sign(self.sigma1, self._slope_strength())


class GevLocationAgainstGumbel(GevTrendTestTwoParameters, GevLocationTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, quantile_level, GumbelTemporalModel, fit_method=fit_method)

    @classproperty
    def label(self):
        return super().label % '\\zeta_0, \\mu_1'

    @classproperty
    def marker(self):
        return 'o'

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4


class GevScaleAgainstGumbel(GevTrendTestTwoParameters, GevScaleTrendTest):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year, quantile_level, GumbelTemporalModel, fit_method=fit_method)

    @classproperty
    def label(self):
        return super().label % '\\zeta_0, \\sigma_1'

    @classproperty
    def marker(self):
        return '^'

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4
