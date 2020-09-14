from typing import List, Dict

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, ticks_values_and_labels_for_percentages, get_half_colormap, ticks_values_and_labels_for_positive_value
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
    get_altitude_group_from_altitudes
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import \
    OneFoldFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AltitudesStudiesVisualizerForNonStationaryModels(StudyVisualizer):

    def __init__(self, studies: AltitudesStudies,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 show=False,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_anderson_test=True,
                 top_n_values_to_remove=None):
        super().__init__(studies.study, show=show, save_to_file=not show)
        self.studies = studies
        self.non_stationary_models = model_classes
        self.fit_method = fit_method
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_anderson_test = display_only_model_that_pass_anderson_test
        self.massif_names = massif_names if massif_names is not None else self.study.all_massif_names()
        self.massif_name_to_massif_id = {m: i for i, m in enumerate(self.massif_names)}
        self.altitude_group = get_altitude_group_from_altitudes(self.studies.altitudes)
        # Load one fold fit
        self._massif_name_to_one_fold_fit = {}
        for massif_name in self.massif_names:
            if any([massif_name in study.study_massif_names for study in self.studies.altitude_to_study.values()]):
                assert top_n_values_to_remove is None
                dataset = studies.spatio_temporal_dataset(massif_name=massif_name,
                                                          top_n_values_to_remove=top_n_values_to_remove)
                old_fold_fit = OneFoldFit(massif_name, dataset, model_classes, self.fit_method,
                                          self.temporal_covariate_for_fit,
                                          type(self.altitude_group),
                                          self.display_only_model_that_pass_anderson_test)
                self._massif_name_to_one_fold_fit[massif_name] = old_fold_fit

    @property
    def massif_name_to_one_fold_fit(self) -> Dict[str, OneFoldFit]:
        return {massif_name: old_fold_fit for massif_name, old_fold_fit in self._massif_name_to_one_fold_fit.items()
                if not self.display_only_model_that_pass_anderson_test
                or old_fold_fit.has_at_least_one_valid_non_stationary_model}

    def plot_moments(self):
        for method_name in ['moment', 'changes_in_the_moment', 'relative_changes_in_the_moment']:
            for order in [1, 2, None]:
                # self.plot_against_years(method_name, order)
                self.plot_map_moment(method_name, order)

    def plot_map_moment(self, method_name, order):
        # Compute values
        massif_name_to_value = {}
        for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items():
            value = one_fold_fit.__getattribute__(method_name)([self.altitude_group.reference_altitude], order=order)[0]
            massif_name_to_value[massif_name] = value

        # Common plot settings
        moment = ' '.join(method_name.split('_'))
        moment = moment.replace('moment', '{} in 2019'.format(OneFoldFit.get_moment_str(order=order)))
        plot_name = '{}{} annual maxima of {}'.format(OneFoldFit.folder_for_plots, moment,
                                                                 SCM_STUDY_CLASS_TO_ABBREVIATION[
                                                                     self.studies.study_class])
        ylabel = '{} ({})'.format(plot_name, self.study.variable_unit)

        # Plot the map
        if any([np.isinf(v) for v in massif_name_to_value.values()]):
            print("shape to large > 0.5, thus removing std that are infinite")
        massif_name_to_value = {m: v for m, v in massif_name_to_value.items()
                                if not np.isinf(v)}

        print(massif_name_to_value)
        negative_and_positive_values = min(massif_name_to_value.values()) < 0
        self.plot_map(cmap=plt.cm.coolwarm, fit_method=self.fit_method, graduation=10,
                      label=ylabel, massif_name_to_value=massif_name_to_value,
                      plot_name=plot_name, add_x_label=True, negative_and_positive_values=negative_and_positive_values,
                      massif_name_to_text=None, altitude=self.altitude_group.reference_altitude)



        # ax.get_xaxis().set_visible(True)
        # ax.set_xticks([])
        # ax.set_xlabel('Altitude = {}m'.format(self.altitude_group.reference_altitude), fontsize=15)


        # cmap = get_shifted_map(min_ratio, max_ratio)
        # print(massif_name_to_value)
        # massif_name_to_color = {m: get_colors([v], cmap, -max_abs_change, max_abs_change)[0]
        #                         for m, v in massif_name_to_value.items()}
        #
        #
        # ticks_values_and_labels = ticks_values_and_labels_for_percentages(graduation, max_abs_change)
        #
        # ax = self.study.visualize_study(massif_name_to_value=massif_name_to_value,
        #                                 replace_blue_by_white=False,
        #                                 axis_off=False, show_label=False,
        #                                 add_colorbar=add_colorbar,
        #                                 # massif_name_to_marker_style=self.massif_name_to_marker_style,
        #                                 # marker_style_to_label_name=self.selected_marker_style_to_label_name,
        #                                 massif_name_to_color=massif_name_to_color,
        #                                 cmap=cmap,
        #                                 show=False,
        #                                 ticks_values_and_labels=ticks_values_and_labels,
        #                                 label=ylabel,
        #                                 add_legend=False,
        #                                 )

        # self.plot_name = plot_name
        # self.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True,
        #                           dpi=500)
        # ax.clear()

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

    def plot_abstract_fast(self, massif_name_to_value, label, graduation=10.0, cmap=plt.cm.coolwarm, add_x_label=True,
                           negative_and_positive_values=True, massif_name_to_text=None):
        plot_name = '{}{}'.format(OneFoldFit.folder_for_plots, label)
        self.plot_map(cmap, self.fit_method, graduation, label, massif_name_to_value, plot_name, add_x_label,
                      negative_and_positive_values,
                      massif_name_to_text)

    @property
    def massif_name_to_shape(self):
        return {massif_name: one_fold_fit.best_shape
                for massif_name, one_fold_fit in self.massif_name_to_one_fold_fit.items()}

    @property
    def massif_name_to_name(self):
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
        self.plot_abstract_fast(massif_name_to_best_coef,
                                label='{}Coef/{} plot for {} {}'.format(OneFoldFit.folder_for_plots,
                                                                         coef_name,
                                                                         SCM_STUDY_CLASS_TO_ABBREVIATION[
                                                                             type(self.study)],
                                                                         self.study.variable_unit),
                                add_x_label=False, graduation=graduation, massif_name_to_text=self.massif_name_to_name,
                                negative_and_positive_values=negative_and_positive_values)

    def plot_shape_map(self):
        self.plot_abstract_fast(self.massif_name_to_shape,
                                label='Shape parameter for {} maxima of {} in 2019 at {}m'.format(
                                    self.study.season_name,
                                    SCM_STUDY_CLASS_TO_ABBREVIATION[
                                        type(self.study)],
                                    self.altitude_group.reference_altitude),
                                add_x_label=False, graduation=0.1, massif_name_to_text=self.massif_name_to_name,
                                cmap=matplotlib.cm.get_cmap('BrBG_r'))

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
