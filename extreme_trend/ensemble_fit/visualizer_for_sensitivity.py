from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import List, Dict

import numpy as np

from extreme_data.meteo_france_data.adamont_data.cmip5.temperature_to_year import get_interval_limits, \
    get_year_min_and_year_max, get_ticks_labels_for_interval
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import get_altitude_class_from_altitudes, \
    get_linestyle_for_altitude_class
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class VisualizerForSensivity(object):

    def __init__(self, altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractSpatioTemporalPolynomialModel],
                 ensemble_fit_classes=None,
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 confidence_interval_based_on_delta_method=False,
                 remove_physically_implausible_models=False,
                 is_temperature_interval=False,
                 is_shift_interval=False,
                 ):
        self.ensemble_fit_classes = ensemble_fit_classes
        self.is_shift_interval = is_shift_interval
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.is_temperature_interval = is_temperature_interval
        self.altitudes_list = altitudes_list
        self.massif_names = massif_names
        self.left_limits, self.right_limits = get_interval_limits(self.is_temperature_interval,
                                                                  self.is_shift_interval)

        self.left_limit_to_right_limit = OrderedDict(zip(self.left_limits, self.right_limits))
        self.right_limit_to_visualizer = {}  # type: Dict[float, VisualizerForProjectionEnsemble]

        for left_limit, right_limit in zip(self.left_limits, self.right_limits):
            interval_str_prefix = "{}-{}".format(left_limit, right_limit)
            interval_str_prefix = interval_str_prefix.replace('.', ',')
            print("Interval is", interval_str_prefix)
            # Build gcm_to_year_min_and_year_max
            gcm_to_year_min_and_year_max = {}
            gcm_list = list(set([g for g, r in gcm_rcm_couples]))
            for gcm in gcm_list:
                year_min_and_year_max = get_year_min_and_year_max(gcm, scenario, left_limit, right_limit,
                                                                  self.is_temperature_interval)
                if year_min_and_year_max[0] is not None:
                    gcm_to_year_min_and_year_max[gcm] = year_min_and_year_max

            print(gcm_to_year_min_and_year_max)
            visualizer = VisualizerForProjectionEnsemble(
                altitudes_list, gcm_rcm_couples, study_class, season, scenario,
                model_classes=model_classes,
                fit_method=fit_method,
                ensemble_fit_classes=ensemble_fit_classes,
                display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                massif_names=massif_names,
                temporal_covariate_for_fit=temporal_covariate_for_fit,
                remove_physically_implausible_models=remove_physically_implausible_models,
                gcm_to_year_min_and_year_max=gcm_to_year_min_and_year_max,
                interval_str_prefix=interval_str_prefix,
            )
            self.right_limit_to_visualizer[right_limit] = visualizer

    def plot(self):
        for visualizer in self.right_limit_to_visualizer.values():
            visualizer.plot()
        merge_visualizer_str_list = []
        if IndependentEnsembleFit in self.ensemble_fit_classes:
            merge_visualizer_str_list.extend([AbstractEnsembleFit.Median_merge, AbstractEnsembleFit.Mean_merge])
        if TogetherEnsembleFit in self.ensemble_fit_classes:
            merge_visualizer_str_list.append(AbstractEnsembleFit.Together_merge)
        for merge_visualizer_str in merge_visualizer_str_list:
            self.sensitivity_plot_percentages(merge_visualizer_str)
            self.sensitivity_plot_return_levels(merge_visualizer_str)
            for relative in [True, False]:
                self.sensitivity_plot_changes(merge_visualizer_str, relative)

    def sensitivity_plot_return_levels(self, merge_visualizer_str):
        ax = plt.gca()
        for altitudes in self.altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
            self.interval_plot_return_levels(ax, altitude_class, merge_visualizer_str)

        ticks_labels = get_ticks_labels_for_interval(self.is_temperature_interval, self.is_shift_interval)
        name = 'Return levels at the end of the interval'
        ax.set_ylabel(name)
        ax.set_xlabel('Interval used to compute the trends ')
        ax.set_xticks(self.right_limits)
        ax.set_xticklabels(ticks_labels)
        lim_left, lim_right = ax.get_xlim()
        ax.legend(prop={'size': 7}, loc='upper center', ncol=2)
        # ax.set_ylim((0, 122))
        # ax.set_yticks([i * 10 for i in range(11)])
        self.save_plot(merge_visualizer_str, name)

    def interval_plot_return_levels(self, ax, altitude_class, merge_visualizer_str):
        linestyle = get_linestyle_for_altitude_class(altitude_class)

        mean_return_levels = []
        for v in self.right_limit_to_visualizer.values():
            merge_visualizer = self.get_merge_visualizer(altitude_class, v, merge_visualizer_str)
            mean_return_level = merge_visualizer.mean_return_level(self.massif_names)
            mean_return_levels.append(mean_return_level)

        label = altitude_class().formula
        ax.plot(self.right_limits, mean_return_levels, linestyle=linestyle, label=label, color='orange')

    def interval_plot_changes(self, ax, altitude_class, merge_visualizer_str, relative):
        linestyle = get_linestyle_for_altitude_class(altitude_class)

        mean_changes = []
        for v in self.right_limit_to_visualizer.values():
            merge_visualizer = self.get_merge_visualizer(altitude_class, v, merge_visualizer_str)
            changes, non_stationary_changes = merge_visualizer.all_changes(self.massif_names, relative=relative,
                                                                           with_significance=False)
            mean_changes.append(np.mean(changes))

        label = altitude_class().formula
        ax.plot(self.right_limits, mean_changes, linestyle=linestyle, label=label, color='darkgreen')

    def sensitivity_plot_changes(self, merge_visualizer_str, relative):
        ax = plt.gca()
        for altitudes in self.altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
            self.interval_plot_changes(ax, altitude_class, merge_visualizer_str, relative)

        ticks_labels = get_ticks_labels_for_interval(self.is_temperature_interval, self.is_shift_interval)
        name = 'relative changes' if relative else "changes"
        ax.set_ylabel(name)
        ax.set_xlabel('Interval used to compute the trends ')
        ax.set_xticks(self.right_limits)
        ax.set_xticklabels(ticks_labels)
        lim_left, lim_right = ax.get_xlim()
        ax.hlines(0, xmin=lim_left, xmax=lim_right)
        ax.legend(prop={'size': 7}, loc='upper center', ncol=2)
        # ax.set_ylim((0, 122))
        # ax.set_yticks([i * 10 for i in range(11)])
        self.save_plot(merge_visualizer_str, name)

    def interval_plot_changes(self, ax, altitude_class, merge_visualizer_str, relative):
        linestyle = get_linestyle_for_altitude_class(altitude_class)

        mean_changes = []
        for v in self.right_limit_to_visualizer.values():
            merge_visualizer = self.get_merge_visualizer(altitude_class, v, merge_visualizer_str)
            changes, non_stationary_changes = merge_visualizer.all_changes(self.massif_names, relative=relative,
                                                                           with_significance=False)
            mean_changes.append(np.mean(changes))

        label = altitude_class().formula
        ax.plot(self.right_limits, mean_changes, linestyle=linestyle, label=label, color='darkgreen')

    def sensitivity_plot_percentages(self, merge_visualizer_str):
        ax = plt.gca()
        for altitudes in self.altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
            self.interval_plot_percentages(ax, altitude_class, merge_visualizer_str)

        ticks_labels = get_ticks_labels_for_interval(self.is_temperature_interval, self.is_shift_interval)
        ax.set_ylabel('Percentages of massifs (\%)')
        ax.set_xlabel('Interval used to compute the trends ')
        ax.set_xticks(self.right_limits)
        ax.set_xticklabels(ticks_labels)
        ax.legend(prop={'size': 7}, loc='upper center', ncol=2)
        ax.set_ylim((0, 122))
        ax.set_yticks([i * 10 for i in range(11)])
        self.save_plot(merge_visualizer_str, "percentages")

    def interval_plot_percentages(self, ax, altitude_class, merge_visualizer_str):
        linestyle = get_linestyle_for_altitude_class(altitude_class)
        increasing_key = 'increasing'
        decreasing_key = 'decreasing'

        label_to_color = {
            increasing_key: 'red',
            decreasing_key: 'blue'
        }
        label_to_l = {
            increasing_key: [],
            decreasing_key: []
        }
        for v in self.right_limit_to_visualizer.values():
            merge_visualizer = self.get_merge_visualizer(altitude_class, v, merge_visualizer_str)
            _, *trends = merge_visualizer.all_trends(self.massif_names, with_significance=False,
                                                     with_relative_change=True)
            label_to_l[decreasing_key].append(trends[0])
            label_to_l[increasing_key].append(trends[2])
        altitude_str = altitude_class().formula
        for label, l in label_to_l.items():
            label_improved = 'with {} trends {}'.format(label, altitude_str)
            color = label_to_color[label]
            ax.plot(self.right_limits, l, label=label_improved, color=color, linestyle=linestyle)

    # Merge visualizer

    def get_merge_visualizer(self, altitude_class, visualizer_projection: VisualizerForProjectionEnsemble,
                             merge_visualizer_str):
        if merge_visualizer_str in [AbstractEnsembleFit.Median_merge, AbstractEnsembleFit.Mean_merge]:
            independent_ensemble_fit = \
                visualizer_projection.altitude_class_to_ensemble_class_to_ensemble_fit[altitude_class][
                    IndependentEnsembleFit]
            merge_visualizer = independent_ensemble_fit.merge_function_name_to_visualizer[merge_visualizer_str]
        else:
            together_ensemble_fit = \
                visualizer_projection.altitude_class_to_ensemble_class_to_ensemble_fit[altitude_class][
                    TogetherEnsembleFit]
            merge_visualizer = together_ensemble_fit.visualizer
        merge_visualizer.studies.study.gcm_rcm_couple = (merge_visualizer_str, "merge")
        return merge_visualizer

    def first_merge_visualizer(self, merge_visualizer_str):
        altitude_class = get_altitude_class_from_altitudes(self.altitudes_list[0])
        visualizer_projection = list(self.right_limit_to_visualizer.values())[0]
        return self.get_merge_visualizer(altitude_class, visualizer_projection, merge_visualizer_str)

    # Utils

    def save_plot(self, merge_visualizer_str, name):
        merge_visualizer = self.first_merge_visualizer(merge_visualizer_str)
        temp_cov = self.temporal_covariate_for_fit is AnomalyTemperatureWithSplineTemporalCovariate
        merge_visualizer.plot_name = 'Sensitivity plot for {} with  ' \
                                     'shift={} temp_interval={}, temp_cov={}'.format(name, self.is_shift_interval,
                                                                                     self.is_temperature_interval,
                                                                                     temp_cov)
        merge_visualizer.show_or_save_to_file(no_title=True)
        plt.close()
