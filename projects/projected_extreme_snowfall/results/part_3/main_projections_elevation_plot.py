import datetime
import time

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import plot_nb_massif_on_upper_axis
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.get_nb_linear_pieces import run_selection, \
    eliminate_massif_name_with_too_much_zeros
from projects.projected_extreme_snowfall.results.part_3.projection_elevation_plot_utils import \
    plot_pychart_scatter_plot, plot_relative_change_at_massif_level, \
    plot_relative_change_at_massif_level_sensitivity_to_frequency
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load
import matplotlib.pyplot as plt

import numpy as np

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()

    fast = False
    snowfall = True
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)
    season = Season.annual

    altitudes = [900, 1500, 2100, 2700, 3300]
    all_massif_names = AbstractStudy.all_massif_names()[:]

    # altitudes = altitudes[-2:]
    # all_massif_names = ['Maurienne', "Mont-Blanc"][:]

    visualizers = []
    for altitude in altitudes:
        print('altitude', altitude)
        altitudes_list = [[altitude]]

        ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]
        # massif_names = ['Mercantour', 'Thabor', 'Devoluy', 'Parpaillon', 'Haut_Var-Haut_Verdon'][:2]

        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
            all_massif_names,
            altitude,
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall)

        massif_name_to_param_name_to_climate_coordinates_with_effects = {}
        for massif_name, parametrization_number in massif_name_to_parametrization_number.items():
            combination = (parametrization_number, parametrization_number, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)
            massif_name_to_param_name_to_climate_coordinates_with_effects[
                massif_name] = param_name_to_climate_coordinates_with_effects

        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, season, scenario,
            model_classes=massif_name_to_model_class,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            fit_method=fit_method,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            safran_study_class=safran_study_class,
            linear_effects=linear_effects,
            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
            param_name_to_climate_coordinates_with_effects=massif_name_to_param_name_to_climate_coordinates_with_effects,
        )

        sub_visualizers = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)]
        print(len(sub_visualizers))
        sub_visualizer = sub_visualizers[0]
        visualizers.append(sub_visualizer)

    # Illustrate the percentage of massifs
    with_significance = False
    covariates = [1.5, 2, 2.5, 3, 3.5, 4][:]
    for with_return_level in [True, False]:
        plot_pychart_scatter_plot(visualizers, all_massif_names, covariates, with_return_level)
        for covariate in covariates:
            print("covariate", covariate)
            OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
            plot_histogram_all_trends_against_altitudes(visualizers, all_massif_names, covariate, with_significance,
                                                        with_return_level)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)

    # Illustrate the trend of each massif
    return_periods = [2, 5, 10, 20, 50, 100]
    with_significance = False
    for relative_change in [True, False]:
        for massif_name in all_massif_names:
            for visualizer in visualizers:
                if massif_name in visualizer.massif_name_to_one_fold_fit:
                    plot_relative_change_at_massif_level_sensitivity_to_frequency(visualizer, massif_name,
                                                                                  with_significance, relative_change,
                                                                                  return_periods)
            plot_relative_change_at_massif_level(visualizers, massif_name, False,
                                                 with_significance, relative_change, None)
            for return_period in return_periods:
                plot_relative_change_at_massif_level(visualizers, massif_name, True,
                                                     with_significance, relative_change, return_period)

def plot_histogram_all_trends_against_altitudes(visualizer_list, massif_names, covariate, with_significance=True,
                                                with_return_level=True):
    assert with_significance is False
    visualizer = visualizer_list[0]

    all_trends = [v.all_trends(massif_names, with_significance=with_significance, with_return_level=with_return_level)
                  for v in visualizer_list]
    nb_massifs, *all_l = zip(*all_trends)

    plt.close()
    ax = plt.gca()
    width = 6
    size = 10
    legend_fontsize = 13
    labelsize = 10
    linewidth = 3
    x = np.array([3 * width * (i + 1) for i in range(len(nb_massifs))])

    colors = ['blue', 'darkblue', 'red', 'darkred']
    # colors = ['red', 'darkred', 'limegreen', 'darkgreen']
    labels = []
    for suffix in ['decrease', 'increase']:
        prefixs = ['Non significant', 'Significant']
        for prefix in prefixs:
            labels.append('{} {}'.format(prefix, suffix))
    for l, color, label in zip(all_l, colors, labels):
        shift = 0.6 * width
        is_a_decrease_plot = colors.index(color) in [0, 1]
        x_shifted = x - shift if is_a_decrease_plot else x + shift
        if with_significance or (not label.startswith("S")):
            if not with_significance:
                label = label.split()[-1]
            ax.bar(x_shifted, l, width=width, color=color, edgecolor=color, label=label,
                   linewidth=linewidth, align='center')
    ax.legend(loc='upper right', prop={'size': size})
    ax.set_ylabel('Percentage of massifs with increasing/decreasing trends\n'
                  'between +1 degree and +{} degrees of global warming (\%)'.format(covariate),
                  fontsize=legend_fontsize)
    ax.set_xlabel('Elevation', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()
    _, ylim_max = ax.get_ylim()
    ax.set_ylim([0, max(ylim_max, 79)])
    ax.set_xticklabels(["{} m".format(v.study.altitude) for v in visualizer_list])

    plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x, range=False)

    label = "return level" if with_return_level else "mean"
    visualizer.plot_name = 'All trends for {} at {} degrees'.format(label, OneFoldFit.COVARIATE_AFTER_TEMPERATURE)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


if __name__ == '__main__':
    main()
