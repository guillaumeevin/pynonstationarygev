from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_trend.trend_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationTemporalModel, NonStationaryScaleTemporalModel, NonStationaryShapeTemporalModel, \
    StationaryTemporalModel
from extreme_fit.distribution.gev.gev_params import GevParams
from root_utils import classproperty


class GevTrendTestOneParameter(AbstractGevTrendTest):

    @property
    def degree_freedom_chi2(self) -> int:
        return 1


class GevVersusGev(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=StationaryTemporalModel,
                         constrained_model_class=StationaryTemporalModel,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    @property
    def is_significant(self) -> bool:
        return False

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 3

    def _slope_strength(self):
        return 0.0


class GevTrendTestOneParameterAgainstStationary(GevTrendTestOneParameter):

    def __init__(self, years, maxima, starting_year, unconstrained_model_class, param_name,
                 quantile_level=EUROCODE_QUANTILE,
                 constrained_model_class=StationaryTemporalModel,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=unconstrained_model_class,
                         quantile_level=quantile_level,
                         constrained_model_class=constrained_model_class,
                         fit_method=fit_method)
        self.param_name = param_name

    @property
    def non_stationary_linear_coef(self):
        return self.get_non_stationary_linear_coef(param_name=self.param_name)

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
        return 4


class GevLocationTrendTest(GevTrendTestOneParameterAgainstStationary):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 constrained_model_class=StationaryTemporalModel,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryLocationTemporalModel,
                         constrained_model_class=constrained_model_class,
                         param_name=GevParams.LOC,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params_last_year.time_derivative_of_return_level(p=self.quantile_level,
                                                                                                 mu1=self.non_stationary_linear_coef)

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return False


class GevScaleTrendTest(GevTrendTestOneParameterAgainstStationary):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 constrained_model_class=StationaryTemporalModel,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryScaleTemporalModel,
                         constrained_model_class=constrained_model_class,
                         param_name=GevParams.SCALE,
                         quantile_level=quantile_level,
                         fit_method=fit_method)

    def _slope_strength(self):
        return self.unconstrained_estimator_gev_params_last_year.time_derivative_of_return_level(
            p=self.quantile_level,
            sigma1=self.non_stationary_linear_coef)

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        sigma1 = self.non_stationary_linear_coef
        return self.same_sign(sigma1, self._slope_strength())


class GevShapeTrendTest(GevTrendTestOneParameterAgainstStationary):

    def __init__(self, years, maxima, starting_year, quantile_level=EUROCODE_QUANTILE,
                 fit_method=MarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year,
                         unconstrained_model_class=NonStationaryShapeTemporalModel,
                         param_name=GevParams.SHAPE,
                         quantile_level=quantile_level,
                         fit_method=fit_method)
