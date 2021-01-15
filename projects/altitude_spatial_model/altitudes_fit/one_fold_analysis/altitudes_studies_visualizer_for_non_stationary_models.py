from collections import Counter
from math import ceil, floor
from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, ticks_values_and_labels_for_percentages, get_half_colormap, ticks_values_and_labels_for_positive_value, \
    get_inverse_colormap, get_cmap_with_inverted_blue_and_green_channels, remove_the_extreme_colors
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION, ALL_ALTITUDES_WITHOUT_NAN
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import \
    get_altitude_group_from_altitudes, HighAltitudeGroup, VeyHighAltitudeGroup, MidAltitudeGroup
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 confidence_interval_based_on_delta_method=False
                 ):
        super().__init__(studies.study, show=show, save_to_file=not show)
        self.studies = studies
        self.non_stationary_models = model_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_test = display_only_model_that_pass_anderson_test
        self.massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        self.massif_name_to_massif_id = {m: i for i, m in enumerate(self.massif_names)}
        self.altitude_group = get_altitude_group_from_altitudes(self.studies.altitudes)
        self.confidence_interval_based_on_delta_method = confidence_interval_based_on_delta_method
        # Load one fold fit
        self.massif_name_to_massif_altitudes = {}
        self._massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            # Load valid massif altitudes
            massif_altitudes = self.get_massif_altitudes(massif_name)
            if self.load_condition(massif_altitudes):
                # Save the massif altitudes only for those who pass the condition
                self.massif_name_to_massif_altitudes[massif_name] = massif_altitudes
                # Load dataset
                dataset = studies.spatio_temporal_dataset(massif_name=massif_name, massif_altitudes=massif_altitudes)
                old_fold_fit = OneFoldFit(massif_name, dataset, model_classes, self.fit_method,
                                          self.temporal_covariate_for_fit,
                                          type(self.altitude_group),
                                          self.display_only_model_that_pass_test,
                                          self.confidence_interval_based_on_delta_method)
                self._massif_name_to_one_fold_fit[massif_name] = old_fold_fit
        # Print number of massif without any validated fit
        massifs_without_any_validated_fit = [massif_name
                                             for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()
                                             if not old_fold_fit.has_at_least_one_valid_model]
        print('Not validated:', len(massifs_without_any_validated_fit), massifs_without_any_validated_fit)
        # Cache
        self._method_name_and_order_to_massif_name_to_value = {}
        self._method_name_and_order_to_max_abs = {}
        self._max_abs_for_shape = None

    moment_names = ['moment', 'changes_of_moment', 'relative_changes_of_moment'][:]
    orders = [1, 2, None][2:]

    def get_massif_altitudes(self, massif_name):
        valid_altitudes = [altitude for altitude, study in self.studies.altitude_to_study.items()
                           if massif_name in study.study_massif_names]
        massif_altitudes = []
        for altitude in valid_altitudes:
            study = self.studies.altitude_to_study[altitude]
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
        at_least_two_altitudes = (len(massif_altitudes) >= 2)
        # return reference_altitude_is_in_altitudes and at_least_two_altitudes
        return at_least_two_altitudes

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return {massif_name: old_fold_fit for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()
                if not self.display_only_model_that_pass_test
                or old_fold_fit.has_at_least_one_valid_model}

    def plot_moments(self):
        for method_name in self.moment_names:
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
            # Store it
            self._method_name_and_order_to_massif_name_to_value[c] = massif_name_to_value
        return self._method_name_and_order_to_massif_name_to_value[c]

    def plot_map_moment(self, method_name, order):
        massif_name_to_value = self.method_name_and_order_to_d(method_name, order)
        # Plot settings
        moment = ' '.join(method_name.split('_'))
        str_for_last_year = ' in 2019'
        moment = moment.replace('moment', '{}{}'.format(OneFoldFit.get_moment_str(order=order), str_for_last_year))
        plot_name = '{}{} '.format(OneFoldFit.folder_for_plots, moment)

        massif_name_to_text = self.massif_name_to_best_name
        if 'change' in method_name:
            plot_name = plot_name.replace(str_for_last_year, '')
            plot_name += ' between {} and {}'.format(2019 - OneFoldFit.nb_years, 2019)
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
            cmap = [plt.cm.coolwarm, plt.cm.bwr, plt.cm.seismic][2]
            cmap = get_inverse_colormap(cmap)
            cmap = get_cmap_with_inverted_blue_and_green_channels(cmap)
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
        # return isinstance(self.altitude_group, (VeyHighAltitudeGroup))
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
        moment = moment.replace('moment', '{} in 2019'.format(OneFoldFit.get_moment_str(order=order)))
        plot_name = '{}Model {} annual maxima of {}'.format(OneFoldFit.folder_for_plots, moment,
                                                            SCM_STUDY_CLASS_TO_ABBREVIATION[self.studies.study_class])
        ax.set_ylabel('{} ({})'.format(plot_name, self.study.variable_unit), fontsize=15)
        ax.set_xlabel('altitudes', fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=13)
        self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True)
        ax.clear()

    # def plot_abstract_fast(self, massif_name_to_value, label, graduation=10.0, cmap=plt.cm.coolwarm, add_x_label=True,
    #                        negative_and_positive_values=True, massif_name_to_text=None):
    #     plot_name = '{}{}'.format(OneFoldFit.folder_for_plots, label)
    #     self.plot_map(cmap, self.fit_method, graduation, label, massif_name_to_value, plot_name, add_x_label,
    #                   negative_and_positive_values,
    #                   massif_name_to_text)

    @property
    def massif_name_to_shape(self):
        return {massif_name: one_fold_fit.best_shape
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    @property
    def massif_name_to_best_name(self):
        return {massif_name: one_fold_fit.best_name
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    def plot_best_coef_maps(self):
        for param_name in GevParams.PARAM_NAMES:
            coordinate_names = [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_T]
            dim_to_coordinate_name = dict(zip([0, 1], coordinate_names))
            for dim in [0, 1, (0, 1)]:
                coordinate_name = LinearCoef.coefficient_name(dim, dim_to_coordinate_name)
                for degree in range(4):
                    coef_name = ' '.join([param_name + coordinate_name + str(degree)])
                    massif_name_to_best_coef = {}
                    for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
                        best_coef = one_fold_fit.best_coef(param_name, dim, degree)
                        if best_coef is not None:
                            massif_name_to_best_coef[massif_name] = best_coef

                    if len(massif_name_to_best_coef) > 0:
                        for evaluate_coordinate in [False, True][:1]:
                            if evaluate_coordinate:
                                coef_name += 'evaluated at coordinates'
                                for massif_name in massif_name_to_best_coef.keys():
                                    if AbstractCoordinates.COORDINATE_X in coordinate_name:
                                        massif_name_to_best_coef[massif_name] *= np.power(1000, degree)
                                    if AbstractCoordinates.COORDINATE_T in coordinate_name:
                                        massif_name_to_best_coef[massif_name] *= np.power(2019, degree)
                            self.plot_best_coef_map(coef_name.replace('_', ''), massif_name_to_best_coef)

    def plot_best_coef_map(self, coef_name, massif_name_to_best_coef):
        values = list(massif_name_to_best_coef.values())
        graduation = (max(values) - min(values)) / 6
        print(coef_name, graduation, max(values), min(values))
        negative_and_positive_values = (max(values) > 0) and (min(values) < 0)
        raise NotImplementedError
        self.plot_map(massif_name_to_value=massif_name_to_best_coef,
                      label='{}Coef/{} plot for {} {}'.format(OneFoldFit.folder_for_plots,
                                                              coef_name,
                                                              SCM_STUDY_CLASS_TO_ABBREVIATION[
                                                                  type(self.study)],
                                                              self.study.variable_unit),
                      add_x_label=False, graduation=graduation, massif_name_to_text=self.massif_name_to_best_name,
                      negative_and_positive_values=negative_and_positive_values)

    def plot_shape_map(self):

        label = 'Shape parameter in 2019 (no unit)'
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
                        gev_params = one_fold_fit.best_function_from_fit.get_params(coordinate, is_transformed=False)
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
            plot_name = '{}{}/Peak year for {}'.format(OneFoldFit.folder_for_plots, peak_year_folder,
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
        margin_function = self.massif_name_to_one_fold_fit[massif_name].best_function_from_fit
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

    def all_trends(self, massif_names):
        """return percents which contain decrease, significant decrease, increase, significant increase percentages"""
        valid_massif_names = self.get_valid_names(massif_names)

        nb_valid_massif_names = len(valid_massif_names)
        nbs = np.zeros(4)
        for one_fold in [one_fold for m, one_fold in self.massif_name_to_one_fold_fit.items()
                         if m in valid_massif_names]:
            # Compute nb of non stationary models
            if one_fold.change_in_return_level_for_reference_altitude == 0:
                continue
            # Compute nbs
            idx = 0 if one_fold.change_in_return_level_for_reference_altitude < 0 else 2
            nbs[idx] += 1
            if one_fold.is_significant:
                nbs[idx + 1] += 1

        percents = 100 * nbs / nb_valid_massif_names
        return [nb_valid_massif_names] + list(percents)

    def all_changes(self, massif_names, relative=False):
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
                if one_fold.is_significant:
                    non_stationary_significant_changes.append(change)

        moment = 'relative mean' if relative else 'Mean'
        print('{} for {}m'.format(moment, self.altitude_group.reference_altitude), np.mean(changes))
        return changes, non_stationary_changes, non_stationary_significant_changes

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
            standard_gumbel_quantiles = one_fold_fit.standard_gumbel_quantiles()
            unconstrained_empirical_quantiles = one_fold_fit.best_estimator.sorted_empirical_standard_gumbel_quantiles()
            all_quantiles = standard_gumbel_quantiles + unconstrained_empirical_quantiles
            epsilon = 0.1
            ax_lim = [min(all_quantiles) - epsilon, max(all_quantiles) + epsilon]

            model_name = self.massif_name_to_best_name[massif_name]
            altitudes = self.massif_name_to_massif_altitudes[massif_name]
            massif_name_corrected = massif_name.replace('_', ' ')
            label = '{} for altitudes  {}'.format(massif_name_corrected, ' & '.join([str(a) + 'm' for a in altitudes]))
            ax.plot(standard_gumbel_quantiles, unconstrained_empirical_quantiles, linestyle='None',
                    label=label + '\n(selected model is ${}$)'.format(model_name), marker='o')

            size_label = 20
            ax.set_xlabel("Theoretical quantile", fontsize=size_label)
            ax.set_ylabel("Empirical quantile", fontsize=size_label)
            ax.set_xlim(ax_lim)
            ax.set_ylim(ax_lim)

            ax.plot(ax_lim, ax_lim, color='k')
            ticks = [i for i in range(ceil(ax_lim[0]), floor(ax_lim[1]) + 1)]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=15)
            plot_name = 'qqplot/{}'.format(massif_name_corrected)
            self.studies.show_or_save_to_file(plot_name=plot_name, show=self.show, no_title=True)
            plt.close()
