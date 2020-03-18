from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property
from scipy.stats import chi2

from experiment.eurocode_data.utils import EUROCODE_QUANTILE, YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    StationaryTemporalModel
from extreme_fit.model.utils import SafeRunException
from root_utils import classproperty
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.utils import load_temporal_coordinates_and_dataset


class AbstractGevTrendTest(object):
    RRunTimeError_TREND = 'R RunTimeError trend'
    nb_years_for_quantile_evolution = 10
    SIGNIFICANCE_LEVEL = 0.05

    def __init__(self, years, maxima, starting_year, unconstrained_model_class,
                 constrained_model_class=StationaryTemporalModel,
                 quantile_level=EUROCODE_QUANTILE,
                 fit_method=TemporalMarginFitMethod.extremes_fevd_mle):
        self.years = years
        self.maxima = maxima
        self.starting_year = starting_year
        self.unconstrained_model_class = unconstrained_model_class
        self.constrained_model_class = constrained_model_class
        self.quantile_level = quantile_level
        self.fit_method = fit_method
        # Load observations, coordinates and datasets
        self.coordinates, self.dataset = load_temporal_coordinates_and_dataset(self.maxima, self.years)
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

    # Likelihood ratio test

    @property
    def is_significant(self) -> bool:
        return self.likelihood_ratio > chi2.ppf(q=1 - self.SIGNIFICANCE_LEVEL, df=self.degree_freedom_chi2)

    @property
    def degree_freedom_chi2(self) -> int:
        raise NotImplementedError

    @classproperty
    def total_number_of_parameters_for_unconstrained_model(cls) -> int:
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
        return self.unconstrained_estimator.function_from_fit.get_coef(gev_param_name,
                                                                       AbstractCoordinates.COORDINATE_T)

    @cached_property
    def unconstrained_estimator_gev_params(self) -> GevParams:
        # Constant parameters correspond to the gev params in 1958
        return self.unconstrained_estimator.function_from_fit.get_gev_params(coordinate=np.array([1958]),
                                                                             is_transformed=False)

    @cached_property
    def constrained_estimator_gev_params(self) -> GevParams:
        # Constant parameters correspond to any gev params
        return self.constrained_estimator.function_from_fit.get_gev_params(coordinate=np.array([1958]),
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
            gev_params = self.unconstrained_estimator.function_from_fit.get_gev_params(
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

    def intensity_plot_wrt_standard_gumbel(self, massif_name, altitude, psnow):
        ax = plt.gca()
        sorted_maxima = sorted(self.maxima)
        label_generic = '{} massif \nat {} m '.format(massif_name, altitude)
        size = 15

        # Plot for the empirical
        standard_gumbel_quantiles = self.get_standard_gumbel_quantiles()
        ax.plot(standard_gumbel_quantiles, sorted_maxima, linestyle='None',
                label='Empirical model', marker='o', color='black')



        ax_twiny = ax.twiny()
        return_periods = [10, 25, 50]
        quantiles = self.get_standard_quantiles_for_return_periods(return_periods, psnow)
        ax_twiny.plot(quantiles, [0 for _ in quantiles], linewidth=0)
        ax_twiny.set_xticks(quantiles)
        ax_twiny.set_xticklabels([return_period for return_period in return_periods])
        ax_twiny.set_xlabel('Return period w.r.t all annual maxima of GSL (years)', fontsize=size)

        # Plot for the selected model with line break
        unconstrained_empirical_quantiles = self.compute_empirical_quantiles(self.unconstrained_estimator)
        # ax.plot(unconstrained_empirical_quantiles, sorted_maxima,
        #         label='Selected model, which is ${}$'.format(self.label))
        # Plot tor the selected model for different year

        end_real_proba = 1 - (0.02 / psnow)
        stationary = True
        if stationary:
            self.plot_model(ax, None, end_proba=end_real_proba, label='Selected model\nwhich is ${}$'.format(self.label),
                            color='grey')
        else:
            self.plot_model(ax, 1959, end_proba=end_real_proba, label='Selected model, which is ${}$'.format(self.label))
            self.plot_model(ax, 2019, end_proba=end_real_proba, label='Selected model, which is ${}$'.format(self.label))

        # Plot for the discarded model
        # if 'Verdon' in massif_name and altitude == 300:
        #     q = [-1.4541688117485054, -1.2811308174310914, -1.216589300814509, -0.7635793791201918, -0.6298883422064275,
        #          -0.5275954855697504, -0.4577268043676126, -0.4497570331795861, -0.1647955002136654,
        #          -0.14492222503785876, -0.139173823298689, -0.11945617994263039, -0.07303100174657867,
        #          -5.497308509286266e-05, 0.13906416388625908, 0.15274793441408543, 0.1717763342727519,
        #          0.17712605315013535, 0.17900143646245203, 0.371986176207554, 0.51640780422156, 0.7380550963951035,
        #          0.7783015252180445, 0.887836077295502, 0.917853338231094, 0.9832396811506262, 1.0359396416309927,
        #          1.1892663813729711, 1.2053261113817888, 1.5695111391491652, 2.3223652143938476, 2.674882764437432,
        #          2.6955728524900406, 2.8155882785356896, 3.282838470153471, 3.2885313947906765]
        #     color = 'red'
        #     ax.plot(q, sorted_maxima,
        #             label='Discarded model, which is ${}$\n'.format('\mathcal{M}_{\zeta_0, \sigma_1}')
        #                   + 'with $\zeta_0=0.84$',
        #             color=color)




        ax_lim = [-1.5, 4]
        ax.set_xlim(ax_lim)
        ax_twiny.set_xlim(ax_lim)
        ax.set_xticks([-1 + i for i in range(6)])
        epsilon = 0.005
        ax.set_ylim(bottom=-epsilon)
        lalsize = 13
        ax.tick_params(axis='both', which='major', labelsize=lalsize)
        ax_twiny.tick_params(axis='both', which='major', labelsize=lalsize)

        ax.yaxis.grid()

        ax.set_xlabel("Standard Gumbel quantile", fontsize=size)
        ax.set_ylabel("Non-zero annual maxima of GSL ({})".format(AbstractSnowLoadVariable.UNIT), fontsize=size)
        ax.legend(prop={'size': 17})

    def plot_model(self, ax, year, start_proba=0.02, end_proba=0.98, label='', color=None):
        standard_gumbel = GevParams(0, 1, 0)
        start_quantile = standard_gumbel.quantile(start_proba)
        end_quantile = standard_gumbel.quantile(end_proba)
        extended_quantiles = np.linspace(start_quantile, end_quantile, 500)
        label = 'Y({})'.format(year) if year is not None else label
        if year is None:
            year = 2019
        gev_params_year = self.unconstrained_estimator.function_from_fit.get_gev_params(
                coordinate=np.array([year]),
                is_transformed=False)
        extended_maxima = [gev_params_year.gumbel_inverse_standardization(q) for q in extended_quantiles]

        ax.plot(extended_quantiles, extended_maxima, linestyle='-', label=label, color=color, linewidth=5)

    def linear_extension(self, ax, q, quantiles, sorted_maxima):
        # Extend the curve linear a bit if the return period 50 is not in the quantiles
        def compute_slope_intercept(x, y):
            x1, x2 = x[-2:]
            y1, y2 = y[-2:]
            a = (y2 - y1) / (x2 - x1)
            b = y1 - a * x1
            return a, b

        def compute_maxima_corresponding_to_return_period(return_period_quantiles, quantiles, model_maxima):
            a, b = compute_slope_intercept(quantiles, model_maxima)
            return a * return_period_quantiles + b

        quantile_return_period_50 = quantiles[-1]
        if max(q) < quantile_return_period_50:
            maxima_extended = compute_maxima_corresponding_to_return_period(quantile_return_period_50,
                                                                            q,
                                                                            sorted_maxima)
            ax.plot([q[-1], quantile_return_period_50],
                    [sorted_maxima[-1], maxima_extended], linestyle='--', label='linear extension')

    def qqplot_wrt_standard_gumbel(self, massif_name, altitude):
        ax = plt.gca()
        size = 15
        standard_gumbel_quantiles = self.get_standard_gumbel_quantiles()
        unconstrained_empirical_quantiles = self.compute_empirical_quantiles(self.unconstrained_estimator)
        constrained_empirical_quantiles = self.compute_empirical_quantiles(self.constrained_estimator)
        all_quantiles = standard_gumbel_quantiles + unconstrained_empirical_quantiles + constrained_empirical_quantiles
        epsilon = 0.5
        ax_lim = [min(all_quantiles) - epsilon, max(all_quantiles) + epsilon]
        ax.plot(standard_gumbel_quantiles, standard_gumbel_quantiles, color='k')
        # ax.plot(standard_gumbel_quantiles, constrained_empirical_quantiles, 'x',
        #         label='Stationary Gumbel model $\mathcal{M}_0$')

        massif_name = massif_name.replace('_', ' ')
        label_generic = '{} massif \nat {} m '.format(massif_name, altitude)
        ax.plot(standard_gumbel_quantiles, unconstrained_empirical_quantiles, linestyle='None',
                label=label_generic + '(selected model is ${}$)'.format(self.label), marker='o')
        if 'Verdon' in massif_name and altitude == 300:
            q = [-1.4541688117485054, -1.2811308174310914, -1.216589300814509, -0.7635793791201918, -0.6298883422064275,
                 -0.5275954855697504, -0.4577268043676126, -0.4497570331795861, -0.1647955002136654,
                 -0.14492222503785876, -0.139173823298689, -0.11945617994263039, -0.07303100174657867,
                 -5.497308509286266e-05, 0.13906416388625908, 0.15274793441408543, 0.1717763342727519,
                 0.17712605315013535, 0.17900143646245203, 0.371986176207554, 0.51640780422156, 0.7380550963951035,
                 0.7783015252180445, 0.887836077295502, 0.917853338231094, 0.9832396811506262, 1.0359396416309927,
                 1.1892663813729711, 1.2053261113817888, 1.5695111391491652, 2.3223652143938476, 2.674882764437432,
                 2.6955728524900406, 2.8155882785356896, 3.282838470153471, 3.2885313947906765]
            print(len(q), len(standard_gumbel_quantiles))
            ax.plot(standard_gumbel_quantiles, q, linestyle='None',
                    label=label_generic
                          + '(discarded model is ${}$\n'.format('\mathcal{M}_{\zeta_0, \sigma_1}')
                          + 'with $\zeta_0=0.84$)',
                    marker='o')

        ax.set_xlabel("Standard Gumbel quantile", fontsize=size)
        ax.set_ylabel("Standard Empirical quantile", fontsize=size)
        ax.legend(loc='lower right', prop={'size': 10})
        ax.set_xlim(ax_lim)
        ax.set_ylim(ax_lim)
        ticks = [i for i in range(ceil(ax_lim[0]), floor(ax_lim[1]) + 1)]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid()
        ax.tick_params(labelsize=size)

        plt.show()

    def get_standard_gumbel_quantiles(self):
        # Standard Gumbel quantiles
        standard_gumbel_distribution = GevParams(loc=0, scale=1, shape=0)
        n = len(self.years)
        standard_gumbel_quantiles = [standard_gumbel_distribution.quantile(i / (n + 1)) for i in range(1, n + 1)]
        return standard_gumbel_quantiles

    def get_standard_quantiles_for_return_periods(self, return_periods, psnow):
        n = len(self.years)
        p_list = [1 - ((1 / return_period) / psnow) for return_period in return_periods]
        standard_gumbel_distribution = GevParams(loc=0, scale=1, shape=0)
        corresponding_quantiles = [standard_gumbel_distribution.quantile(p) for p in p_list]
        return corresponding_quantiles

    def compute_empirical_quantiles(self, estimator):
        empirical_quantiles = []
        for year, maximum in sorted(zip(self.years, self.maxima), key=lambda t: t[1]):
            gev_param = estimator.function_from_fit.get_gev_params(
                coordinate=np.array([year]),
                is_transformed=False)
            maximum_standardized = gev_param.gumbel_standardization(maximum)
            empirical_quantiles.append(maximum_standardized)
        return empirical_quantiles

    # For some visualizations

    def return_level_plot_comparison(self, ax, label, color=None):
        # ax = plt.gca()
        size = 15
        # Load Gev parameter for the year of interest for the unconstrained estimator
        gev_params, gev_params_with_corrected_shape = self.get_gev_params_with_big_shape_and_correct_shape()
        suffix = 'in {}'.format(YEAR_OF_INTEREST_FOR_RETURN_LEVEL)
        gev_params.return_level_plot_against_return_period(ax, color, linestyle='-', label=label,
                                                           suffix_return_level_label=suffix)
        gev_params_with_corrected_shape.return_level_plot_against_return_period(ax, color=color, linestyle='--',
                                                                                suffix_return_level_label=suffix)

    def return_level_plot_difference(self, ax, ax2, label, color=None):
        gev_params, gev_params_with_corrected_shape = self.get_gev_params_with_big_shape_and_correct_shape()
        return_periods = list(range(2, 61))
        quantile_with_big_shape = gev_params.get_return_level(return_periods)
        quantile_with_small_shape = gev_params_with_corrected_shape.get_return_level(return_periods)
        difference_quantile = quantile_with_big_shape - quantile_with_small_shape
        relative_difference = 100 * difference_quantile / quantile_with_small_shape
        # Plot the difference on the two axis

        ax.vlines(50, 0, np.max(difference_quantile))
        ax.plot(return_periods, difference_quantile, color=color, linestyle='-', label=label)
        ax.legend(loc='upper left')
        difference_ylabel = 'difference return level in 2019'
        ax.set_ylabel(difference_ylabel + ' (kPa)')

        ax2.vlines(50, 0, np.max(relative_difference))
        ax2.plot(return_periods, relative_difference, color=color, linestyle='--', label=label)
        ax2.legend(loc='upper right')
        ax2.yaxis.grid()
        ax2.set_ylabel('relative ' + difference_ylabel + ' (%)')

        ax.set_xlabel('Return period')
        ax.set_xticks([10 * i for i in range(1, 7)])
        plt.gca().set_ylim(bottom=0)

    def get_gev_params_with_big_shape_and_correct_shape(self):
        gev_params = self.unconstrained_estimator.function_from_fit.get_gev_params(coordinate=np.array([YEAR_OF_INTEREST_FOR_RETURN_LEVEL]),
                                                                                   is_transformed=False)  # type: GevParams
        gev_params_with_corrected_shape = GevParams(loc=gev_params.location,
                                                    scale=gev_params.scale,
                                                    shape=0.5)
        return gev_params, gev_params_with_corrected_shape
