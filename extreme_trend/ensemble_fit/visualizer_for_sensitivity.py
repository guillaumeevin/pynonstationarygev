from collections import OrderedDict
import matplotlib.pyplot as plt
from typing import List, Dict

from extreme_data.meteo_france_data.adamont_data.cmip5.temperature_to_year import get_temp_min_and_temp_max, \
    temperature_minmax_to_year_minmax, get_ticks_labels_for_temp_min_and_temp_max
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.polynomial_margin_model.spatio_temporal_polynomial_model import \
    AbstractSpatioTemporalPolynomialModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.elevation_temporal_model_for_projections.independent_ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit
from extreme_trend.elevation_temporal_model_for_projections.visualizer_for_projection_ensemble import \
    VisualizerForProjectionEnsemble
from extreme_trend.one_fold_analysis.altitude_group import get_altitude_class_from_altitudes, \
    get_linestyle_for_altitude_class


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
                 merge_visualizer_str=IndependentEnsembleFit.Median_merge # if we choose the Mean merge, then it is almost impossible to obtain stationary trends
                 ):
        self.merge_visualizer_str = merge_visualizer_str
        self.altitudes_list = altitudes_list
        self.massif_names = massif_names
        self.temp_min, self.temp_max = get_temp_min_and_temp_max()
        self.temp_min_to_temp_max = OrderedDict(zip(self.temp_min, self.temp_max))
        self.temp_min_to_visualizer = {} # type: Dict[float, VisualizerForProjectionEnsemble]

        for temp_min, temp_max in zip(self.temp_min, self.temp_max):
            print(temp_min, temp_max)
            # Build 
            gcm_to_year_min_and_year_max = {}
            gcm_list = list(set([g for g, r in gcm_rcm_couples]))
            for gcm in gcm_list:
                year_min_and_year_max = temperature_minmax_to_year_minmax(gcm, scenario, temp_min, temp_max)
                if year_min_and_year_max[0] is not None:
                    gcm_to_year_min_and_year_max[gcm] = year_min_and_year_max
                
            visualizer = VisualizerForProjectionEnsemble(
                altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
                model_classes=model_classes,
                fit_method=fit_method,
                ensemble_fit_classes=ensemble_fit_classes,
                display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                confidence_interval_based_on_delta_method=confidence_interval_based_on_delta_method,
                massif_names=massif_names,
                temporal_covariate_for_fit=temporal_covariate_for_fit,
                remove_physically_implausible_models=remove_physically_implausible_models,
                gcm_to_year_min_and_year_max=gcm_to_year_min_and_year_max
            )
            self.temp_min_to_visualizer[temp_min] = visualizer

    def plot(self):
        # todo: before reactivating the subplot, i should ensure that we can modify the prefix
        # so that we can have all the subplot for the merge visualizer
        # , and not just the plots for the last t_min
        # for visualizer in self.temp_min_to_visualizer.values():
        #     visualizer.plot()
        self.sensitivity_plot()

    def sensitivity_plot(self):
        ax = plt.gca()
        for altitudes in self.altitudes_list:
            altitude_class = get_altitude_class_from_altitudes(altitudes)
            self.temperature_interval_plot(ax, altitude_class)

        ticks_labels = get_ticks_labels_for_temp_min_and_temp_max()
        ax.set_ylabel('Percentages of massifs (\%)')
        ax.set_xlabel('Range of temperatures used to compute the trends ')
        ax.set_xticks(self.temp_min)
        ax.set_xticklabels(ticks_labels)
        ax.legend(prop={'size': 7}, loc='upper center', ncol=2)
        ax.set_ylim((0, 122))
        ax.set_yticks([i*10 for i in range(11)])
        merge_visualizer = self.first_merge_visualizer
        merge_visualizer.plot_name = 'Sensitivity plot'
        merge_visualizer.show_or_save_to_file(no_title=True)

    @property
    def first_merge_visualizer(self):
        altitude_class = get_altitude_class_from_altitudes(self.altitudes_list[0])
        visualizer_projection = list(self.temp_min_to_visualizer.values())[0]
        return self.get_merge_visualizer(altitude_class, visualizer_projection)

    def get_merge_visualizer(self, altitude_class, visualizer_projection: VisualizerForProjectionEnsemble):
        independent_ensemble_fit = visualizer_projection.altitude_class_to_ensemble_class_to_ensemble_fit[altitude_class][
            IndependentEnsembleFit]
        merge_visualizer = independent_ensemble_fit.merge_function_name_to_visualizer[self.merge_visualizer_str]
        merge_visualizer.studies.study.gcm_rcm_couple = (self.merge_visualizer_str, "merge")
        return merge_visualizer

    def temperature_interval_plot(self, ax, altitude_class):
        linestyle = get_linestyle_for_altitude_class(altitude_class)
        increasing_key = 'increasing'
        decreasing_key = 'decreasing'
        label_to_l = {
            increasing_key: [],
            decreasing_key: []
        }
        label_to_color = {
            increasing_key: 'red',
            decreasing_key: 'blue'
        }
        for v in self.temp_min_to_visualizer.values():
            merge_visualizer = self.get_merge_visualizer(altitude_class, v)
            _, *trends = merge_visualizer.all_trends(self.massif_names, with_significance=False)
            label_to_l[decreasing_key].append(trends[0])
            label_to_l[increasing_key].append(trends[2])
        altitude_str = altitude_class().formula
        for label, l in label_to_l.items():
            label_improved = 'with {} trends {}'.format(label, altitude_str)
            color = label_to_color[label]
            ax.plot(self.temp_min, l, label=label_improved, color=color, linestyle=linestyle)

