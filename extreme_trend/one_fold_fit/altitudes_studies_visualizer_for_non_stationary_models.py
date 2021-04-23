from collections import Counter
from math import ceil, floor
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_inverse_colormap, \
    remove_the_extreme_colors
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.altitude_group import \
    get_altitude_group_from_altitudes, VeyHighAltitudeGroup, MidAltitudeGroup
from extreme_trend.one_fold_fit.one_fold_fit import \
    OneFoldFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):
    consider_at_least_two_altitudes = True

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 climate_coordinates_with_effects=None
                 ):
        super().__init__(studies.study, show=show, save_to_file=not show)
        self.studies = studies
        self.model_classes = model_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_test = display_only_model_that_pass_anderson_test
        self.massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        self.massif_name_to_massif_id = {m: i for i, m in enumerate(self.massif_names)}
        self.altitude_group = get_altitude_group_from_altitudes(self.studies.altitudes)
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.climate_coordinates_with_effects = climate_coordinates_with_effects

        self.massif_name_to_massif_altitudes = {}
        # Load one fold fit
        self.load_one_fold_fit()

        # Cache
        self._method_name_and_order_to_massif_name_to_value = {}
        self._method_name_and_order_to_max_abs = {}
        self._max_abs_for_shape = None

    def load_one_fold_fit(self):
        one_fold_fit_list = [self.fit_one_fold(massif_name) for massif_name in self.massif_names]
        self._massif_name_to_one_fold_fit = {m: o for m, o in zip(self.massif_names, one_fold_fit_list) if
                                             o is not None}
        # Print number of massif without any validated fit
        massifs_without_any_validated_fit = [massif_name
                                             for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()
                                             if not old_fold_fit.has_at_least_one_valid_model]
        print('Not validated:', len(massifs_without_any_validated_fit), massifs_without_any_validated_fit)

    def fit_one_fold(self, massif_name):
        # Load valid massif altitudes
        massif_altitudes = self.get_massif_altitudes(massif_name)
        if self.load_condition(massif_altitudes):
            # Save the massif altitudes only for those who pass the condition
            self.massif_name_to_massif_altitudes[massif_name] = massif_altitudes
            # Load dataset
            dataset = self.get_dataset(massif_altitudes, massif_name)
            old_fold_fit = OneFoldFit(massif_name, dataset, self.model_classes,
                                      self.study.year_min,
                                      self.study.year_max,
                                      self.fit_method,
                                      self.temporal_covariate_for_fit,
                                      self.altitude_group,
                                      self.display_only_model_that_pass_test,
                                      self.confidence_interval_based_on_delta_method,
                                      self.remove_physically_implausible_models,
                                      self.climate_coordinates_with_effects)
            return old_fold_fit
        else:
            return None

    def get_dataset(self, massif_altitudes, massif_name):
        dataset = self.studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes)
        return dataset

    moment_names = ['moment', 'changes_of_moment', 'relative_changes_of_moment'][:]
    orders = [1, 2, None][2:]

    def get_massif_altitudes(self, massif_name):
        return self._get_massif_altitudes(massif_name, self.studies)

    def _get_massif_altitudes(self, massif_name, studies):
        valid_altitudes = [altitude for altitude, study in studies.altitude_to_study.items()
                           if massif_name in study.study_massif_names]
        massif_altitudes = []
        for altitude in valid_altitudes:
            study = studies.altitude_to_study[altitude]
            annual_maxima = study.massif_name_to_annual_maxima[massif_name]
            percentage_of_non_zeros = 100 * np.count_nonzero(annual_maxima) / len(annual_maxima)
            if percentage_of_non_zeros > 90:
                massif_altitudes.append(altitude)
            # else:
            #     print(massif_name, altitude, percentage_of_non_zeros)
        return massif_altitudes

    def load_condition(self, massif_altitudes):
        # At least two altitudes for the estimated
        # reference_altitude_is_in_altitudes = (self.altitude_group.reference_altitude in massif_altitudes)
        if self.consider_at_least_two_altitudes:
            return len(massif_altitudes) >= 2
        else:
            return True

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return {massif_name: old_fold_fit for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()
                if old_fold_fit.has_at_least_one_valid_model}

    @property
    def first_one_fold_fit(self):
        return list(self.massif_name_to_one_fold_fit.values())[0]

    def plot_moments(self):
        for method_name in self.moment_names[:2]:
            for order in self.orders:
                # self.plot_against_years(method_name, order)
                self.plot_map_moment(method_name, order)

    def method_name_and_order_to_max_abs(self, method_name, order):
        c = (method_name, order)
        if c not in self._method_name_and_order_to_max_abs:
            return None
        else:
            return self._method_name_and_order_to_max_abs[c]

    def method_name_and_order_to_d(self, method_name, order):
        c = (method_name, order)
        if c not in self._method_name_and_order_to_massif_name_to_value:
            # Compute values
            massif_name_to_value = {}
            for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
                value = \
                    one_fold_fit.__getattribute__(method_name)([self.altitude_group.reference_altitude], order=order)[0]
                massif_name_to_value[massif_name] = value
            # Remove values
            if any([np.isinf(v) for v in massif_name_to_value.values()]):
                print("shape to large > 0.5, thus removing std that are infinite")
            massif_name_to_value = {m: v for m, v in massif_name_to_value.items()
                                    if not np.isinf(v)}
            # todo: i could remove here potential undefined parameters
            # Store it
            self._method_name_and_order_to_massif_name_to_value[c] = massif_name_to_value
        return self._method_name_and_order_to_massif_name_to_value[c]

    def ratio_groups(self):
        return [self.ratio_uncertainty_interval_size(altitude, self.first_one_fold_fit.last_year) for altitude in
                self.studies.altitudes]

    def ratio_uncertainty_interval_size(self, altitude, year):
        study = self.studies.altitude_to_study[altitude]
        massif_name_to_interval = study.massif_name_to_stationary_gev_params_and_confidence(OneFoldFit.quantile_level,
                                                                                            self.confidence_interval_based_on_delta_method)[
            1]
        massif_names_with_pointwise_interval = set(massif_name_to_interval)
        valid_massif_names = set(self.massif_name_to_one_fold_fit.keys())
        intersection_massif_names = valid_massif_names.intersection(massif_names_with_pointwise_interval)
        ratios = []
        for massif_name in intersection_massif_names:
            one_fold_fit = self.massif_name_to_one_fold_fit[massif_name]
            new_interval_size = one_fold_fit.best_confidence_interval(altitude, year).interval_size
            old_interval_size = massif_name_to_interval[massif_name].interval_size
            ratio = new_interval_size / old_interval_size
            ratios.append(ratio)
        return ratios

    def plot_map_moment(self, method_name, order):
        massif_name_to_value = self.method_name_and_order_to_d(method_name, order)
        # Plot settings
        moment = ' '.join(method_name.split('_'))
        d_temperature = {'C': '{C}'}
        str_for_last_year = ' at +${}^o\mathrm{C}$' \
            if self.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate else ' in {}'
        str_for_last_year = str_for_last_year.format(self.first_one_fold_fit.covariate_after, **d_temperature)
        moment = moment.replace('moment', '{}{}'.format(OneFoldFit.get_moment_str(order=order), str_for_last_year))
        plot_name = '{} '.format(moment)

        if 'change' in method_name:
            plot_name = plot_name.replace(str_for_last_year, '')
            plot_name += self.first_one_fold_fit.between_covariate_str

            if 'relative' not in method_name:
                # Put the relative score as text on the plot for the change.
                massif_name_to_text = {m: ('+' if v > 0 else '') + str(int(v)) + '\%' for m, v in
                                       self.method_name_and_order_to_d(self.moment_names[2], order).items()}

        parenthesis = self.study.variable_unit if 'relative' not in method_name else '\%'
        ylabel = '{} ({})'.format(plot_name, parenthesis)

        max_abs_change = self.method_name_and_order_to_max_abs(method_name, order)
        add_colorbar = self.add_colorbar

        is_return_level_plot = (self.moment_names.index(method_name) == 0) and (order is None)
        if is_return_level_plot:

            cmap = plt.cm.Spectral
            cmap = remove_the_extreme_colors(cmap, epsilon=0.25)
            cmap = get_inverse_colormap(cmap)
            add_colorbar = True
            max_abs_change = None
            massif_name_to_text = {m: round(v) for m, v in massif_name_to_value.items()}
            graduation = self.altitude_group.graduation_for_return_level
            fontsize_label = 17
        else:
            # cmap = plt.cm.RdYlGn
            cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][1]
            # cmap = get_inverse_colormap(cmap)
            # cmap = get_cmap_with_inverted_blue_and_green_channels(cmap)
            cmap = remove_the_extreme_colors(cmap)
            graduation = 10
            fontsize_label = 10

        negative_and_positive_values = self.moment_names.index(method_name) > 0
        # Plot the map

        self.plot_map(cmap=cmap, graduation=graduation,
                      label=ylabel, massif_name_to_value=massif_name_to_value,
                      plot_name=plot_name, add_x_label=True,
                      negative_and_positive_values=negative_and_positive_values,
                      altitude=self.altitude_group.reference_altitude,
                      add_colorbar=add_colorbar,
                      max_abs_change=max_abs_change,
                      massif_name_to_text=massif_name_to_text,
                      xlabel=self.altitude_group.xlabel,
                      fontsize_label=fontsize_label,
                      )

    @property
    def add_colorbar(self):
        # return True
        return isinstance(self.altitude_group, (VeyHighAltitudeGroup, MidAltitudeGroup))

    def plot_against_years(self, method_name, order):
        ax = plt.gca()
        min_altitude, *_, max_altitude = self.studies.altitudes
        altitudes_plot = np.linspace(min_altitude, max_altitude, num=50)
        for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
            massif_altitudes = self.studies.massif_name_to_altitudes[massif_name]
            ind = (min(massif_altitudes) <= altitudes_plot) & (altitudes_plot <= max(massif_altitudes))
            massif_altitudes_plot = altitudes_plot[ind]
            values = one_fold_fit.__getattribute__(method_name)(massif_altitudes_plot, order=order)
            massif_id = self.massif_name_to_massif_id[massif_name]
            plot_against_altitude(massif_altitudes_plot, ax, massif_id, massif_name, values)
        # Plot settings
        ax.legend(prop={'size': 7}, ncol=3)
        moment = ' '.join(method_name.split('_'))
        moment = moment.replace('moment',
                                '{} in {}'.format(OneFoldFit.get_moment_str(order=order), self.first_one_fold_fit.last_year))
        plot_name = 'Model {} annual maxima of {}'.format(moment,
                                                          SCM_STUDY_CLASS_TO_ABBREVIATION[self.studies.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True)
        ax.clear()

    @property
    def massif_name_to_shape(self):
        return {massif_name: one_fold_fit.best_shape
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    @property
    def massif_name_to_best_name(self):
        return {massif_name: one_fold_fit.best_name
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    def plot_shape_map(self):

        label = 'Shape parameter in {} (no unit)'.format(self.first_one_fold_fit.last_year)
        max_abs_change = self._max_abs_for_shape + 0.05
        self.plot_map(massif_name_to_value=self.massif_name_to_shape,
                      label=label,
                      plot_name=label,
                      fontsize_label=15,
                      add_x_label=True, graduation=0.1,
                      massif_name_to_text=self.massif_name_to_best_name,
                      cmap=matplotlib.cm.get_cmap('BrBG_r'),
                      altitude=self.altitude_group.reference_altitude,
                      add_colorbar=self.add_colorbar,
                      max_abs_change=max_abs_change,
                      xlabel=self.altitude_group.xlabel,
                      )

    def plot_altitude_for_the_peak(self):
        pass

    def plot_year_for_the_peak(self, plot_mean=True):
        t_list = self.study.ordered_years
        return_period = 50
        for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
            ax = plt.gca()
            # One plot for each altitude
            altitudes = np.arange(500, min(3000, max(self.studies.altitudes)), 500)
            for altitude in altitudes:
                i = 0
                while self.studies.altitudes[i] < altitude:
                    i += 1
                nearest_altitude = self.studies.altitudes[i]
                nearest_study = self.studies.altitude_to_study[nearest_altitude]
                if massif_name in nearest_study.study_massif_names:
                    y_list = []
                    for t in t_list:
                        coordinate = np.array([altitude, t])
                        gev_params = one_fold_fit.best_margin_function_from_fit.get_params(coordinate, is_transformed=False)
                        if plot_mean:
                            y = gev_params.mean
                        else:
                            y = gev_params.return_level(return_period=return_period)
                        y_list.append(y)
                    label = '{} m'.format(altitude)
                    ax.plot(t_list, y_list, label=label)
            ax.legend()
            # Modify the limits of the y axis
            lim_down, lim_up = ax.get_ylim()
            ax_lim = (0, lim_up)
            ax.set_ylim(ax_lim)
            ax.set_xlabel('Year')
            if plot_mean:
                ylabel = 'Mean {} maxima'.format(self.study.season_name)
            else:
                ylabel = '{}-year return level'.format(return_period)
            ax.set_ylabel('{} of {} in {} ({})'.format(ylabel, SCM_STUDY_CLASS_TO_ABBREVIATION[type(self.study)],
                                                       massif_name.replace('_', ' '), self.study.variable_unit))
            peak_year_folder = 'Peak year ' + ylabel
            plot_name = '{}/Peak year for {}'.format(peak_year_folder,
                                                     massif_name.replace('_', ''))
            self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True, tight_layout=True)
            plt.close()

    # Plots "altitude switch" and "peak year"

    @property
    def massif_name_to_is_decreasing_parabol(self):
        # For the test we only activate the Mont-Blanc massif
        d = {massif_name: False for massif_name in self.massif_name_to_one_fold_fit.keys()}
        if max(self.study.ordered_years) < 2030:
            for massif_name in ['Vanoise', 'Aravis', 'Beaufortain', 'Chablais']:
                d[massif_name] = True
        return d

    @property
    def massif_name_to_altitudes_switch_and_peak_years(self):
        return {massif_name: self.compute_couple_peak_year_and_altitude_switch(massif_name)
                for massif_name, is_decreasing_parabol in self.massif_name_to_is_decreasing_parabol.items()
                if is_decreasing_parabol}

    def compute_couple_peak_year_and_altitude_switch(self, massif_name):
        # Get the altitude limits
        altitudes = self.study.massif_name_to_altitudes[massif_name]
        # use a step of 100m for instance
        step = 10
        altitudes = list(np.arange(min(altitudes), max(altitudes) + step, step))
        # Get all the correspond peak years
        margin_function = self.massif_name_to_one_fold_fit[massif_name].best_margin_function_from_fit
        peak_years = []
        year_left = 1900
        switch_altitudes = []
        for altitude in altitudes:
            year_left = self.compute_peak_year(margin_function, altitude, year_left)
            if year_left > 2020:
                break
            peak_years.append(year_left)
            switch_altitudes.append(altitude)
        print(switch_altitudes)
        print(peak_years)
        return switch_altitudes, peak_years

    def compute_peak_year(self, margin_function: AbstractMarginFunction, altitude, year_left):
        year_right = year_left + 0.1
        mean_left = margin_function.get_params(np.array([altitude, year_left])).mean
        mean_right = margin_function.get_params(np.array([altitude, year_right])).mean
        print(year_left, year_right, mean_left, mean_right)
        if mean_right < mean_left:
            return year_left
        else:
            return self.compute_peak_year(margin_function, altitude, year_right)

    def plot_peak_year_against_altitude(self):
        ax = plt.gca()
        for massif_name, (altitudes, peak_years) in self.massif_name_to_altitudes_switch_and_peak_years.items():
            ax.plot(altitudes, peak_years, label=massif_name)
        ax.legend()
        ax.set_xlabel('Altitude')
        ax.set_ylabel('Peak years')
        plot_name = 'Peak Years'
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show)
        plt.close()

    def plot_altitude_switch_against_peak_year(self):
        ax = plt.gca()
        for massif_name, (altitudes, peak_years) in self.massif_name_to_altitudes_switch_and_peak_years.items():
            ax.plot(peak_years, altitudes, label=massif_name)
        ax.legend()
        ax.set_xlabel('Peak years')
        ax.set_ylabel('Altitude')
        plot_name = 'Switch altitude'
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show)
        plt.close()

    def massif_name_to_return_level(self, massif_names):
        valid_massif_names = self.get_valid_names(massif_names)
        massif_name_to_return_level = {}
        for m, one_fold in self.massif_name_to_one_fold_fit.items():
            if m in valid_massif_names:
                massif_name_to_return_level[m] = one_fold.return_level_last_temporal_coordinate
        return massif_name_to_return_level

    def all_trends(self, massif_names, with_significance=True, with_relative_change=False):
        """return percents which contain decrease, significant decrease, increase, significant increase percentages"""
        valid_massif_names = self.get_valid_names(massif_names)

        nb_valid_massif_names = len(valid_massif_names)
        nbs = np.zeros(4)
        for one_fold in [one_fold for m, one_fold in self.massif_name_to_one_fold_fit.items()
                         if m in valid_massif_names]:
            # Compute nb of non stationary models
            if with_relative_change:
                change_value = one_fold.relative_change_in_return_level_for_reference_altitude
            else:
                change_value = one_fold.change_in_return_level_for_reference_altitude
            if change_value == 0:
                continue
            # Compute nbs
            idx = 0 if change_value < 0 else 2
            nbs[idx] += 1
            if with_significance and one_fold.is_significant:
                nbs[idx + 1] += 1

        percents = 100 * nbs / nb_valid_massif_names
        return [nb_valid_massif_names] + list(percents)

    def all_changes(self, massif_names, relative=False, with_significance=True):
        """return percents which contain decrease, significant decrease, increase, significant increase percentages"""
        valid_massif_names = self.get_valid_names(massif_names)
        changes = []
        non_stationary_changes = []
        non_stationary_significant_changes = []
        for one_fold in [one_fold for m, one_fold in self.massif_name_to_one_fold_fit.items()
                         if m in valid_massif_names]:
            # Compute changes
            if relative:
                change = one_fold.relative_change_in_return_level_for_reference_altitude
            else:
                change = one_fold.change_in_return_level_for_reference_altitude
            changes.append(change)
            if change != 0:
                non_stationary_changes.append(change)
                if with_significance:
                    if one_fold.is_significant:
                        non_stationary_significant_changes.append(change)

        moment = 'relative mean' if relative else 'Mean'
        print('{} for {}m'.format(moment, self.altitude_group.reference_altitude), np.mean(changes))
        if with_significance:
            return changes, non_stationary_changes, non_stationary_significant_changes
        else:
            return changes, non_stationary_changes

    def get_valid_names(self, massif_names):
        valid_massif_names = set(self.massif_name_to_one_fold_fit.keys())
        if massif_names is not None:
            valid_massif_names = valid_massif_names.intersection(set(massif_names))
        return valid_massif_names

    def model_name_to_percentages(self, massif_names, only_significant=False):
        valid_massif_names = self.get_valid_names(massif_names)
        nb_valid_massif_names = len(valid_massif_names)
        best_names = [one_fold_fit.best_estimator.margin_model.name_str
                      for m, one_fold_fit in self.massif_name_to_one_fold_fit.items()
                      if m in valid_massif_names and (not only_significant or one_fold_fit.is_significant)]
        counter = Counter(best_names)
        d = {name: 100 * c / nb_valid_massif_names for name, c in counter.items()}
        # Add 0 for the name not present
        for name in self.model_names:
            if name not in d:
                d[name] = 0
        return d

    @property
    def model_names(self):
        massif_name = list(self.massif_name_to_one_fold_fit.keys())[0]
        return self.massif_name_to_one_fold_fit[massif_name].model_names

    def plot_qqplots(self):
        for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
            ax = plt.gca()
            altitudes = self.massif_name_to_massif_altitudes[massif_name]
            massif_name_corrected = massif_name.replace('_', ' ')

            all_quantiles = []

            for altitude in self.studies.altitudes:
                coordinate_for_filter = (altitude, None)
                # We filter on the transformed gumbel quantiles for the altitude of interest
                unconstrained_empirical_quantiles = one_fold_fit.best_estimator.sorted_empirical_standard_gumbel_quantiles(
                    coordinate_for_filter=coordinate_for_filter)
                n = len(unconstrained_empirical_quantiles)
                if n > 0:
                    assert n == 61
                    standard_gumbel_quantiles = one_fold_fit.standard_gumbel_quantiles(n=n)
                    ax.plot(standard_gumbel_quantiles, unconstrained_empirical_quantiles, linestyle='None',
                            label='{} m'.format(altitude), marker='o')

                    all_quantiles.extend(standard_gumbel_quantiles)
                    all_quantiles.extend(unconstrained_empirical_quantiles)

            size_label = 20
            ax.set_xlabel("Theoretical quantile", fontsize=size_label)
            ax.set_ylabel("Empirical quantile", fontsize=size_label)

            epsilon = 0.1
            ax_lim = [min(all_quantiles) - epsilon, max(all_quantiles) + epsilon]
            ax.set_xlim(ax_lim)
            ax.set_ylim(ax_lim)

            ax.plot(ax_lim, ax_lim, color='k')
            ticks = [i for i in range(ceil(ax_lim[0]), floor(ax_lim[1]) + 1)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            labelsize = 15
            ax.tick_params(labelsize=labelsize)
            plot_name = 'qqplot/{}'.format(massif_name_corrected)
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles[::-1], labels[::-1], prop={'size': labelsize})
            self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True)
            plt.close()
