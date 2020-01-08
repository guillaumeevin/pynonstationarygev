import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cached_property import cached_property
from scipy.stats import chi2

from experiment.eurocode_data.utils import EUROCODE_QUANTILE
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from experiment.trend_analysis.univariate_test.utils import load_temporal_coordinates_and_dataset, \
    fitted_linear_margin_estimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel, TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    StationaryTemporalModel
from extreme_fit.model.utils import SafeRunException
from extreme_fit.distribution.gev.gev_params import GevParams
from root_utils import classproperty
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractGevTrendTest(AbstractUnivariateTest):
    RRunTimeError_TREND = 'R RunTimeError trend'
    nb_years_for_quantile_evolution = 10

    def __init__(self, years, maxima, starting_year, unconstrained_model_class,
                 constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE,
                 fit_method=TemporalMarginFitMethod.extremes_fevd_mle):
        super().__init__(years, maxima, starting_year)
        self.unconstrained_model_class = unconstrained_model_class
        self.constrained_model_class = constrained_model_class
        self.quantile_level = quantile_level
        self.fit_method = fit_method
        # Load observations, coordinates and datasets
        self.coordinates, self.dataset = load_temporal_coordinates_and_dataset(maxima, years)
        # By default crashed boolean is False
        self.crashed = False
        try:
            pass
        except SafeRunException:
            self.crashed = True

    @cached_property
    def constrained_estimator(self):
        try:
            return fitted_linear_margin_estimator(self.constrained_model_class, self.coordinates, self.dataset,
                                                  self.starting_year, self.fit_method)
        except SafeRunException:
            self.crashed = True

    @cached_property
    def unconstrained_estimator(self):
        try:
            return fitted_linear_margin_estimator(self.unconstrained_model_class, self.coordinates, self.dataset,
                                                  self.starting_year, self.fit_method)
        except SafeRunException:
            self.crashed = True

    # Type of trends

    @classmethod
    def real_trend_types(cls):
        return super().real_trend_types() + [cls.RRunTimeError_TREND]

    @classmethod
    def get_real_trend_types(cls, display_trend_type):
        real_trend_types = super().get_real_trend_types(display_trend_type)
        if display_trend_type is cls.NON_SIGNIFICATIVE_TREND:
            real_trend_types.append(cls.RRunTimeError_TREND)
        return real_trend_types

    @property
    def test_trend_type(self) -> str:
        if self.crashed:
            return self.RRunTimeError_TREND
        else:
            return super().test_trend_type

    # Likelihood ratio test

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=self.degree_freedom_chi2)

    @property
    def degree_freedom_chi2(self) -> int:
        raise NotImplementedError

    @property
    def total_number_of_parameters_for_unconstrained_model(self) -> int:
        raise NotImplementedError

    @property
    def aic(self):
        # deviance = - 2 * nllh
        return 2 * self.total_number_of_parameters_for_unconstrained_model - self.unconstrained_model_deviance

    @property
    def likelihood_ratio(self):
        return self.unconstrained_model_deviance - self.constrained_model_deviance

    @property
    def constrained_model_deviance(self):
        if self.crashed:
            return np.nan
        else:
            return self.constrained_estimator.result_from_model_fit.deviance

    @property
    def unconstrained_model_deviance(self):
        unconstrained_estimator = self.unconstrained_estimator
        if self.crashed:
            return np.nan
        else:
            return unconstrained_estimator.result_from_model_fit.deviance

    @property
    def unconstained_nllh(self):
        unconstrained_estimator = self.unconstrained_estimator
        if self.crashed:
            return np.nan
        else:
            return unconstrained_estimator.result_from_model_fit.nllh

    # Evolution of the GEV parameters and corresponding quantiles

    @property
    def test_sign(self) -> int:
        return np.sign(self.time_derivative_of_return_level)

    def get_non_stationary_linear_coef(self, gev_param_name: str):
        return self.unconstrained_estimator.margin_function_from_fit.get_coef(gev_param_name,
                                                                              AbstractCoordinates.COORDINATE_T)

    @cached_property
    def unconstrained_estimator_gev_params(self) -> GevParams:
        # Constant parameters correspond to the gev params in 1958
        return self.unconstrained_estimator.margin_function_from_fit.get_gev_params(coordinate=np.array([1958]),
                                                                                    is_transformed=False)

    @cached_property
    def constrained_estimator_gev_params(self) -> GevParams:
        # Constant parameters correspond to any gev params
        return self.constrained_estimator.margin_function_from_fit.get_gev_params(coordinate=np.array([1958]),
                                                                                  is_transformed=False)

    def time_derivative_times_years(self, nb_years):
        # Compute the slope strength
        slope = self._slope_strength()
        # Delta T must in the same unit as were the parameter of slope mu1 and sigma1
        slope *= nb_years * self.coordinates.transformed_distance_between_two_successive_years[0]
        return slope

    @property
    def time_derivative_of_return_level(self):
        if self.crashed:
            return 0.0
        else:
            return self.time_derivative_times_years(self.nb_years_for_quantile_evolution)

    def relative_change_in_return_level(self, initial_year, final_year):
        return_level_values = []
        for year in [initial_year, final_year]:
            gev_params = self.unconstrained_estimator.margin_function_from_fit.get_gev_params(
                coordinate=np.array([year]),
                is_transformed=False)
            return_level_values.append(gev_params.quantile(self.quantile_level))
        change_until_final_year = self.time_derivative_times_years(nb_years=final_year - initial_year)
        change_in_between = (return_level_values[1] - return_level_values[0])
        np.testing.assert_almost_equal(change_until_final_year, change_in_between, decimal=5)
        initial_return_level = return_level_values[0]
        return 100 * change_until_final_year / initial_return_level

    def _slope_strength(self):
        raise NotImplementedError

    @staticmethod
    def same_sign(a, b):
        return (a > 0 and b > 0) or (a < 0 and b < 0)

    @property
    def mean_difference_same_sign_as_slope_strenght(self) -> bool:
        return False

    @property
    def variance_difference_same_sign_as_slope_strenght(self) -> bool:
        return False

    def mean_difference(self, zeta0: float, mu1: float = 0.0, sigma1: float = 0.0) -> float:
        return GevParams(loc=mu1, scale=sigma1, shape=zeta0, accept_zero_scale_parameter=True).mean

    @property
    def test_trend_constant_quantile(self):
        if self.crashed:
            return 0.0
        else:
            return self.unconstrained_estimator_gev_params.quantile(p=self.quantile_level)

    # Some class properties for display purpose

    @classproperty
    def marker(self):
        raise NotImplementedError

    @classproperty
    def label(self):
        return '\\mathcal{M}_{%s}'

    # Some display function

    def qqplot_wrt_standard_gumbel(self, marker, color=None):
        # Standard Gumbel quantiles
        standard_gumbel_distribution = GevParams(loc=0, scale=1, shape=0)
        n = len(self.years)
        standard_gumbel_quantiles = [standard_gumbel_distribution.quantile(i / (n + 1)) for i in range(1, n + 1)]
        unconstrained_empirical_quantiles = self.compute_empirical_quantiles(self.unconstrained_estimator)
        constrained_empirical_quantiles = self.compute_empirical_quantiles(self.constrained_estimator)
        plt.plot(standard_gumbel_quantiles, standard_gumbel_quantiles, color=color)
        plt.plot(standard_gumbel_quantiles, constrained_empirical_quantiles, 'x')
        plt.plot(standard_gumbel_quantiles, unconstrained_empirical_quantiles, linestyle='None', **marker)
        plt.show()

    def compute_empirical_quantiles(self, estimator):
        empirical_quantiles = []
        for year, maximum in sorted(zip(self.years, self.maxima), key=lambda t: t[1]):
            gev_param = estimator.margin_function_from_fit.get_gev_params(
                coordinate=np.array([year]),
                is_transformed=False)
            maximum_standardized = gev_param.gumbel_standardization(maximum)
            empirical_quantiles.append(maximum_standardized)
        return empirical_quantiles
