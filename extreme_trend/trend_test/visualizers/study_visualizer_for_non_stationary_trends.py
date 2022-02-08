from collections import OrderedDict, Counter
from enum import Enum
from multiprocessing.pool import Pool
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from cached_property import cached_property

from extreme_data.eurocode_data.eurocode_region import C2, C1, E
from extreme_data.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE, EUROCODE_RETURN_LEVEL_STR, \
    YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, ticks_values_and_labels_for_percentages
from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
from projects.past_extreme_ground_snow_loads.utils import NON_STATIONARY_TREND_TEST_PAPER_1, ALTITUDE_TO_GREY_MASSIF
from extreme_trend.trend_test.abstract_gev_trend_test import AbstractGevTrendTest
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import GumbelTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    compute_eurocode_confidence_interval, EurocodeConfidenceIntervalFromExtremes
from root_utils import NB_CORES


class ModelSubsetForUncertainty(Enum):
    stationary_gumbel = 0
    stationary_gumbel_and_gev = 1
    non_stationary_gumbel = 2
    non_stationary_gumbel_and_gev = 3
    stationary_gev = 4


class StudyVisualizerForNonStationaryTrends(StudyVisualizer):

    def __init__(self, study: AbstractStudy, show=True, save_to_file=False, only_one_graph=False, only_first_row=False,
                 vertical_kde_plot=False, year_for_kde_plot=None, plot_block_maxima_quantiles=False,
                 temporal_non_stationarity=False, transformation_class=None, verbose=False, multiprocessing=False,
                 complete_non_stationary_trend_analysis=False,
                 uncertainty_methods=None,
                 model_subsets_for_uncertainty=None,
                 uncertainty_massif_names=None,
                 effective_temporal_covariate=YEAR_OF_INTEREST_FOR_RETURN_LEVEL,
                 relative_change_trend_plot=True,
                 non_stationary_trend_test_to_marker=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 select_only_acceptable_shape_parameter=True,
                 fit_gev_only_on_non_null_maxima=False,
                 fit_only_time_series_with_ninety_percent_of_non_null_values=True,
                 select_only_model_that_pass_anderson_test=False,
                 ):
        super().__init__(study, show, save_to_file, only_one_graph, only_first_row, vertical_kde_plot,
                         year_for_kde_plot, plot_block_maxima_quantiles, temporal_non_stationarity,
                         transformation_class, verbose, multiprocessing, complete_non_stationary_trend_analysis)
        # Add some attributes
        self.select_only_model_that_pass_anderson_test = select_only_model_that_pass_anderson_test
        self.fit_only_time_series_with_ninety_percent_of_non_null_values = fit_only_time_series_with_ninety_percent_of_non_null_values
        self.fit_gev_only_on_non_null_maxima = fit_gev_only_on_non_null_maxima
        self.select_only_acceptable_shape_parameter = select_only_acceptable_shape_parameter
        self.fit_method = fit_method
        self.non_stationary_trend_test_to_marker = non_stationary_trend_test_to_marker
        self.relative_change_trend_plot = relative_change_trend_plot
        self.effective_temporal_covariate = effective_temporal_covariate
        self.model_subsets_for_uncertainty = model_subsets_for_uncertainty
        self.uncertainty_methods = uncertainty_methods
        self.uncertainty_massif_names = uncertainty_massif_names
        # Assign some default arguments
        if self.model_subsets_for_uncertainty is None:
            # We regroup the results either with all the models,
            # or only the stationary gumbel model which correspond to the building standards
            self.model_subsets_for_uncertainty = [ModelSubsetForUncertainty.stationary_gumbel,
                                                  ModelSubsetForUncertainty.non_stationary_gumbel_and_gev][:]
        assert isinstance(self.model_subsets_for_uncertainty, list)
        if self.uncertainty_methods is None:
            self.uncertainty_methods = [ConfidenceIntervalMethodFromExtremes.my_bayes,
                                        ConfidenceIntervalMethodFromExtremes.ci_mle][1:]
        if self.uncertainty_massif_names is None:
            self.uncertainty_massif_names = self.study.study_massif_names
        else:
            assert set(self.uncertainty_massif_names).issubset(set(self.study.study_massif_names))
        if self.non_stationary_trend_test_to_marker is None:
            # Assign default argument for the non stationary trends
            self.non_stationary_trend_test = NON_STATIONARY_TREND_TEST_PAPER_1
            self.non_stationary_trend_test_to_marker = {t: t.marker for t in self.non_stationary_trend_test}
        else:
            self.non_stationary_trend_test = list(self.non_stationary_trend_test_to_marker.keys())
        self.global_max_abs_change = None

    # Utils

    @cached_property
    def massif_name_to_years_and_maxima(self):
        d = {}
        df_maxima = self.study.observations_annual_maxima.df_maxima_gev.loc[self.uncertainty_massif_names]
        years = np.array(df_maxima.columns)
        for massif_name, s_maxima in df_maxima.iterrows():
            d[massif_name] = (years, np.array(s_maxima))
        return d

    @cached_property
    def massif_name_to_psnow(self):
        return {m: np.count_nonzero(maxima) / len(maxima) for m, (_, maxima) in
                self.massif_name_to_years_and_maxima.items()}

    @property
    def massifs_names_with_year_without_snow(self):
        return [m for m, psnow in self.massif_name_to_psnow.items() if psnow < 1]

    @cached_property
    def massif_name_to_eurocode_quantile_level_in_practice(self):
        """Due to missing data, the the eurocode quantile which 0.98 if we have all the data
        correspond in practice to the quantile psnow x 0.98 of the data where there is snow"""
        if self.fit_gev_only_on_non_null_maxima:
            return {m: 1 - ((1 - EUROCODE_QUANTILE) / p_snow) for m, p_snow in self.massif_name_to_psnow.items()}
        else:
            return {m: EUROCODE_QUANTILE for m in self.massif_name_to_psnow.keys()}

    @property
    def intersection_of_massif_names_fitted(self) -> List[str]:
        all_massif = set(self.study.all_massif_names())
        missing_massifs_given = set(ALTITUDE_TO_GREY_MASSIF[self.altitude])
        if ModelSubsetForUncertainty.non_stationary_gumbel_and_gev in self.model_subsets_for_uncertainty:
            # Check the assertion for the missing massifs
            # For the paper 1, I should enter in this loop only for the ground snow load
            # (because i should not use this ModelSubsetForUncertainty for the Eurocode)
            massifs_found = set(self.massif_name_to_trend_test_that_minimized_aic.keys())
            missing_massifs_found = all_massif - massifs_found
            assert missing_massifs_found == missing_massifs_given
        # Return by default this result
        intersection_massif_names = all_massif - missing_massifs_given
        return list(intersection_massif_names)

    @cached_property
    def massif_name_to_years_and_maxima_for_model_fitting(self):
        if self.fit_gev_only_on_non_null_maxima:
            d = {}
            for m, (years, maxima) in self.massif_name_to_years_and_maxima.items():
                mask = np.nonzero(maxima)
                d[m] = (years[mask], maxima[mask])
        else:
            d = self.massif_name_to_years_and_maxima
        # In both cases, we remove any massif with psnow < 0.9
        if self.fit_only_time_series_with_ninety_percent_of_non_null_values:
            d = {m: v for m, v in d.items() if self.massif_name_to_psnow[m] >= 0.9}
        return d

    @cached_property
    def massif_name_to_trend_test_that_minimized_aic(self) -> Dict[str, AbstractGevTrendTest]:
        massif_name_to_trend_test_that_minimized_aic = {}
        for massif_name in self.massif_name_to_years_and_maxima_for_model_fitting.keys():
            # Compute sorted trend test
            sorted_trend_test = self.get_sorted_trend_test(massif_name)
            # Extract the stationary or non-stationary model that minimized AIC
            trend_test_that_minimized_aic = sorted_trend_test[0]
            if (not self.select_only_model_that_pass_anderson_test) or \
                    trend_test_that_minimized_aic.goodness_of_fit_anderson_test:
                massif_name_to_trend_test_that_minimized_aic[massif_name] = trend_test_that_minimized_aic
            else:
                print('anderson test was not passed')

        return massif_name_to_trend_test_that_minimized_aic

    def get_trend_trend_test(self, massif_name, trend_test_classes):
        x, y = self.massif_name_to_years_and_maxima_for_model_fitting[massif_name]
        quantile_level = self.massif_name_to_eurocode_quantile_level_in_practice[massif_name]
        all_trend_test = [
            t(years=x, maxima=y, starting_year=None, quantile_level=quantile_level,
              fit_method=self.fit_method)
            for t in trend_test_classes]  # type: List[AbstractGevTrendTest]
        return all_trend_test

    def get_sorted_trend_test(self, massif_name):
        # Compute trend test
        all_trend_test = self.get_trend_trend_test(massif_name, self.non_stationary_trend_test)
        # Exclude GEV models whose shape parameter is not in the support of the prior distribution for GMLE
        if self.select_only_acceptable_shape_parameter:
            acceptable_shape_parameter = lambda s: -0.5 <= s <= 0.5  # physically acceptable prior
            all_trend_test = [t for t in all_trend_test
                              if (acceptable_shape_parameter(t.unconstrained_estimator_gev_params_last_year.shape)
                                  and acceptable_shape_parameter(
                            t.unconstrained_estimator_gev_params_first_year.shape))]
        sorted_trend_test = sorted(all_trend_test, key=lambda t: t.aic)
        return sorted_trend_test

    # Part 1 - Trends

    @property
    def max_abs_change(self):
        return max(abs(min(self.massif_name_to_change_value.values())), max(self.massif_name_to_change_value.values()))

    @cached_property
    def _max_abs_change(self):
        max_abs_change = self.global_max_abs_change if self.global_max_abs_change is not None else self.max_abs_change
        if max_abs_change == 0:
            max_abs_change = 1e-10
        return max_abs_change

    def plot_trends(self, max_abs_tdrl=None, add_colorbar=True):
        if max_abs_tdrl is not None:
            self.global_max_abs_change = max_abs_tdrl
        massif_name_to_text = {m: ('+' if v > 0 else '') + str(int(v)) + '\%' for m, v in self.massif_name_to_change_value.items()}
        print(self.label)
        label = self.label.replace('GSL', 'snow load')
        ax = self.study.visualize_study(massif_name_to_value=self.massif_name_to_change_value,
                                        replace_blue_by_white=False,
                                        axis_off=False, show_label=False,
                                        add_colorbar=add_colorbar,
                                        massif_name_to_marker_style=None,
                                        marker_style_to_label_name=self.selected_marker_style_to_label_name,
                                        massif_name_to_color=self.massif_name_to_color,
                                        cmap=self.cmap,
                                        show=False,
                                        ticks_values_and_labels=self.ticks_values_and_labels,
                                        label=label,
                                        add_legend=False,
                                        massif_name_to_text=massif_name_to_text,
                                        add_text=True)
        ax.get_xaxis().set_visible(True)
        ax.set_xticks([])
        ax.set_xlabel('Elevation = {} m'.format(self.study.altitude), fontsize=15)
        middle_word = 'o' if (not add_colorbar and self.study.altitude in [1800, 2700]) else ''
        self.plot_name = 'tdlr_trends_w' + middle_word + '_colorbar'
        self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True,
                                  dpi=500)
        plt.close()

    @cached_property
    def all_marker_style_to_label_name(self):
        return OrderedDict([(t.marker, t.label) for t in self.non_stationary_trend_test])

    @cached_property
    def selected_marker_style_to_label_name(self):
        marker_style_selected = set([d['marker'] for d in self.massif_name_to_marker_style.values()])
        marker_style_to_label_name = {m: l for m, l in self.all_marker_style_to_label_name.items()
                                      if m in marker_style_selected}
        return marker_style_to_label_name

    @property
    def label(self):
        if self.relative_change_trend_plot:
            label_tdlr_bar = 'Relative change between {} and {}'.format(self.initial_year, self.final_year)
        else:
            label_tdlr_bar = self.label_tdrl_bar
        label = label_tdlr_bar + '\nfor {}'.format(EUROCODE_RETURN_LEVEL_STR)
        if self.relative_change_trend_plot:
            # change units
            label = label.split('(')[0] + '(\%)'
        return label

    @property
    def label_tdrl_bar(self):
        nb_years = AbstractGevTrendTest.nb_years_for_quantile_evolution
        suffix = 'per decade' if nb_years == 10 else 'in {} years'.format(nb_years)
        return 'Change {}'.format(suffix)

    @property
    def ticks_values_and_labels(self):
        return ticks_values_and_labels_for_percentages(self.graduation, self._max_abs_change)

    @property
    def graduation(self):
        graduation = 10 if self.relative_change_trend_plot else 0.2
        return graduation

    @cached_property
    def massif_name_to_tdrl_value(self):
        return {m: t.time_derivative_of_return_level for m, t in
                self.massif_name_to_trend_test_that_minimized_aic.items()}

    @cached_property
    def massif_name_to_relative_change_value(self):
        return {m: t.relative_change_in_return_level(initial_year=self.initial_year, final_year=self.final_year)
                for m, t in self.massif_name_to_trend_test_that_minimized_aic.items()}

    @property
    def initial_year(self):
        return self.final_year - 50

    @property
    def final_year(self):
        return 2010

    @cached_property
    def massif_name_to_change_value(self):
        if self.relative_change_trend_plot:
            return self.massif_name_to_relative_change_value
        else:
            return self.massif_name_to_tdrl_value

    @cached_property
    def cmap(self):
        return get_shifted_map(-self._max_abs_change, self._max_abs_change)

    @cached_property
    def massif_name_to_color(self):
        return {m: get_colors([v], self.cmap, -self._max_abs_change, self._max_abs_change)[0]
                for m, v in self.massif_name_to_change_value.items()}

    @cached_property
    def selected_trend_test_class_counter(self):
        return Counter([type(t) for t in self.massif_name_to_trend_test_that_minimized_aic.values()])

    @cached_property
    def selected_and_significative_trend_test_class_counter(self):
        return Counter(
            [type(t) for t in self.massif_name_to_trend_test_that_minimized_aic.values() if t.is_significant])

    @cached_property
    def selected_and_anderson_goodness_of_fit_trend_test_class_counter(self):
        return Counter(
            [type(t) for t in self.massif_name_to_trend_test_that_minimized_aic.values()
             if t.goodness_of_fit_anderson_test])

    @cached_property
    def selected_and_kstest_goodness_of_fit_trend_test_class_counter(self):
        return Counter(
            [type(t) for t in self.massif_name_to_trend_test_that_minimized_aic.values()
             if t.goodness_of_fit_ks_test])

    @cached_property
    def massif_name_to_marker_style(self):
        d = {}
        for m, t in self.massif_name_to_trend_test_that_minimized_aic.items():
            d[m] = {'marker': self.non_stationary_trend_test_to_marker[type(t)],
                    'color': 'k',
                    'markersize': 7,
                    'fillstyle': 'full' if t.is_significant else 'none'}
        return d

    # Part 2 - Uncertainty return level plot

    def massif_name_and_model_subset_to_model_class(self, massif_name, model_subset_for_uncertainty):
        if model_subset_for_uncertainty is ModelSubsetForUncertainty.stationary_gumbel:
            return GumbelTemporalModel
        elif model_subset_for_uncertainty is ModelSubsetForUncertainty.non_stationary_gumbel_and_gev:
            return self.massif_name_to_trend_test_that_minimized_aic[massif_name].unconstrained_model_class
        else:
            raise ValueError(model_subset_for_uncertainty)

    def all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class(self,
                                                                              ci_method=ConfidenceIntervalMethodFromExtremes.ci_mle,
                                                                              model_subset_for_uncertainty=ModelSubsetForUncertainty.non_stationary_gumbel_and_gev) \
            -> Dict[str, EurocodeConfidenceIntervalFromExtremes]:
        # Compute for the uncertainty massif names
        massifs_names = self.intersection_of_massif_names_fitted
        arguments = [
            [self.massif_name_to_years_and_maxima_for_model_fitting[m],
             self.massif_name_and_model_subset_to_model_class(m, model_subset_for_uncertainty),
             ci_method, self.effective_temporal_covariate,
             self.massif_name_to_eurocode_quantile_level_in_practice[m]
             ] for m in massifs_names]
        if self.multiprocessing:
            with Pool(NB_CORES) as p:
                res = p.starmap(compute_eurocode_confidence_interval, arguments)
        else:
            res = [compute_eurocode_confidence_interval(*argument) for argument in arguments]
        massif_name_to_eurocode_return_level_uncertainty = dict(zip(massifs_names, res))
        # For the rest of the massif names. Create a Eurocode Return Level Uncertainty as nan
        for massif_name in set(self.study.all_massif_names()) - set(massifs_names):
            massif_name_to_eurocode_return_level_uncertainty[massif_name] = self.default_eurocode_uncertainty
        return massif_name_to_eurocode_return_level_uncertainty

    @cached_property
    def default_eurocode_uncertainty(self):
        return EurocodeConfidenceIntervalFromExtremes(mean_estimate=np.nan, confidence_interval=(np.nan, np.nan))

    @cached_property
    def triplet_to_eurocode_uncertainty(self):
        d = {}
        for ci_method in self.uncertainty_methods:
            for model_subset_for_uncertainty in self.model_subsets_for_uncertainty:
                for massif_name, eurocode_uncertainty in self.all_massif_name_to_eurocode_uncertainty_for_minimized_aic_model_class(
                        ci_method, model_subset_for_uncertainty).items():
                    d[(ci_method, model_subset_for_uncertainty, massif_name)] = eurocode_uncertainty
        return d

    def model_name_to_uncertainty_method_to_ratio_above_eurocode(self):
        assert self.uncertainty_massif_names == self.study.study_massif_names

    # Some values for the histogram

    @cached_property
    def massif_name_to_eurocode_values(self):
        """Eurocode values for the altitude"""
        return {m: r().valeur_caracteristique(altitude=self.study.altitude)
                for m, r in massif_name_to_eurocode_region.items() if m in self.uncertainty_massif_names}

    def excess_metrics(self, ci_method, model_subset_for_uncertainty):
        triplet = [(massif_name_to_eurocode_region[massif_name],
                    self.massif_name_to_eurocode_values[massif_name],
                    self.triplet_to_eurocode_uncertainty[(ci_method, model_subset_for_uncertainty, massif_name)])
                   for massif_name in self.intersection_of_massif_names_fitted]
        # First array for histogram
        a = 100 * np.array([(uncertainty.confidence_interval[0] > eurocode,
                             uncertainty.mean_estimate > eurocode,
                             uncertainty.confidence_interval[1] > eurocode)
                            for _, eurocode, uncertainty in triplet])
        a = np.mean(a, axis=0)
        # Second array for curve
        b = [[] for _ in range(3)]
        for eurocode_region, eurocode, uncertainty in triplet:
            diff = uncertainty.mean_estimate - eurocode
            b[0].append(diff)
            if eurocode_region in [C1, C2]:
                b[1].append(diff)
            if eurocode_region in [E]:
                b[2].append(diff)

        b = [np.mean(np.array(e)) for e in b]
        # Third array for curve of percentage of exceedance for return levels
        c = [[] , [],  []]
        for _, eurocode, uncertainty in triplet:
            metrics = [uncertainty.confidence_interval[0], uncertainty.mean_estimate, uncertainty.confidence_interval[1]]
            for j, metric in enumerate(metrics):
                relative_difference = 100 * (metric - eurocode) / eurocode
                c[j].append(relative_difference)
        c = np.array([np.mean(np.array(e)) if len(e) > 0 else 0 for e in c])
        return a, b, c

    # Part 3 - QQPLOT

    def intensity_plot(self, massif_name, color=None):
        psnow = self.massif_name_to_psnow[massif_name]
        trend_test = self.massif_name_to_trend_test_that_minimized_aic[massif_name]
        trend_test.intensity_plot_wrt_standard_gumbel_simple(massif_name, self.altitude, color, psnow)
        self.plot_name = 'intensity_plot_{}_{}_{}_{}'.format(self.altitude, massif_name, psnow,
                                                             trend_test.unconstrained_estimator_gev_params_last_year.shape)
        self.show_or_save_to_file(add_classic_title=False, no_title=True)
        plt.close()

    def qqplot(self, massif_name, color=None):
        trend_test = self.massif_name_to_trend_test_that_minimized_aic[massif_name]
        trend_test.qqplot_wrt_standard_gumbel(massif_name, self.altitude)
        self.plot_name = 'qpplot_plot_{}_{}_{}'.format(self.altitude, massif_name,
                                                       trend_test.unconstrained_estimator_gev_params_last_year.shape)
        self.show_or_save_to_file(add_classic_title=False, no_title=True, tight_layout=True)
        plt.close()

    def return_level_plot(self, ax, ax2, massif_name, color=None):
        trend_test = self.massif_name_to_trend_test_that_minimized_aic[massif_name]
        model_name = '${}$'.format(trend_test.label)
        label = '{} at {}m with {}'.format(massif_name, self.altitude, model_name)
        trend_test.return_level_plot_difference(ax, ax2, label, color)

    # Part 4 - Trend plot

    @property
    def altitude(self):
        return self.study.altitude

    def trend_summary_labels(self):
        return 'altitude', ''

    def trend_summary_values(self):
        trend_tests = list(self.massif_name_to_trend_test_that_minimized_aic.values())
        decreasing_trend_tests = [t for t in trend_tests if t.time_derivative_of_return_level < 0]
        percentage_decrease = 100 * len(decreasing_trend_tests) / len(trend_tests)
        significative_decrease_trend_tests = [t for t in decreasing_trend_tests if t.is_significant]
        percentage_decrease_significative = 100 * len(significative_decrease_trend_tests) / len(trend_tests)
        # For visualization at 2700m
        if percentage_decrease_significative == percentage_decrease:
            percentage_decrease += 0.4
        compute_mean_decrease = lambda l: -np.mean(np.array(list(l)))
        mean_decreases = [compute_mean_decrease(self.massif_name_to_relative_change_value.values())]
        # Compute mean relatives per regions (for the moment i don't add the region means)
        # massif_name_to_region_name = AbstractExtendedStudy.massif_name_to_region_name
        # for region_name in AbstractExtendedStudy.real_region_names:
        #     change_values = [v for m, v in self.massif_name_to_relative_change_value.items()
        #                      if massif_name_to_region_name[m] == region_name]
        #     mean_decreases.append(compute_mean_decrease(change_values))
        return (self.altitude, percentage_decrease, percentage_decrease_significative, *mean_decreases)

    def trend_summary_contrasting_values(self, all_regions=False):
        # trend_tests = list(self.massif_name_to_trend_test_that_minimized_aic.values())
        # decreasing_trend_tests = [t for t in trend_tests if t.time_derivative_of_return_level < 0]
        # percentage_decrease = 100 * len(decreasing_trend_tests) / len(trend_tests)
        # significative_decrease_trend_tests = [t for t in decreasing_trend_tests if t.is_significant]
        # percentage_decrease_significative = 100 * len(significative_decrease_trend_tests) / len(trend_tests)
        compute_mean_change = lambda l: np.mean(np.array(list(l))) if len(l) > 0 else 0
        massif_name_to_region_name = AbstractExtendedStudy.massif_name_to_region_name
        if all_regions:
            mean_changes = [compute_mean_change(self.massif_name_to_relative_change_value.values())]
            # Compute mean relatives per regions (for the moment i don't add the region means)
            for region_name in AbstractExtendedStudy.real_region_names:
                change_values = [v for m, v in self.massif_name_to_relative_change_value.items()
                                 if massif_name_to_region_name[m] == region_name]
                mean_changes.append(compute_mean_change(change_values))
        else:
            mean_changes = [
                compute_mean_change([v for m, v in self.massif_name_to_relative_change_value.items()
                                     if massif_name_to_region_name[m] in regions])
                for regions in
                [AbstractExtendedStudy.real_region_names[:2], AbstractExtendedStudy.real_region_names[2:]]
            ]

        return (self.altitude, *mean_changes)

    def mean_percentage_of_standards_for_massif_names_with_years_without_snow(self):
        model_subset_for_uncertainty = ModelSubsetForUncertainty.non_stationary_gumbel_and_gev
        ci_method = ConfidenceIntervalMethodFromExtremes.ci_mle
        percentages = []
        for massif_name in self.massifs_names_with_year_without_snow:
            eurocode_value = self.massif_name_to_eurocode_values[massif_name]
            eurocode_uncertainty = self.triplet_to_eurocode_uncertainty[
                (ci_method, model_subset_for_uncertainty, massif_name)]
            percentage = 100 * np.array(eurocode_uncertainty.triplet) / eurocode_value
            percentages.append(percentage)
        return np.round(np.mean(percentages, axis=0))

    @property
    def massif_name_to_relative_change_in_psnow(self):
        def compute_relative_change_in_psnow(maxima):
            maxima_before, maxima_after = maxima[:30], maxima[30:]
            psnow_before, psnow_after = [np.count_nonzero(s) / len(s) for s in [maxima_before, maxima_after]]
            return 100 * (psnow_after - psnow_before) / psnow_before

        return {m: compute_relative_change_in_psnow(self.massif_name_to_years_and_maxima[m][1]) for m in
                self.massifs_names_with_year_without_snow}
