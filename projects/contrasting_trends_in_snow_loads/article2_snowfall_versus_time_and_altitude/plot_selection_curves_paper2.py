from typing import Dict
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from extreme_trend.abstract_gev_trend_test import AbstractGevTrendTest
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues
from projects.exceeding_snow_loads.section_results.plot_selection_curves import merge_counter, \
    get_ordered_list_from_counter, permute
from projects.exceeding_snow_loads.utils import dpi_paper1_figure, get_trend_test_name
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import StudyVisualizerForNonStationaryTrends


def plot_selection_curves_paper2(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    visualizer = list(altitude_to_visualizer.values())[0]

    ax = create_adjusted_axes(1, 1)

    vs = [v for v in altitude_to_visualizer.values()]

    selected_counter = merge_counter([v.selected_trend_test_class_counter for v in vs])
    selected_and_anderson_counter = merge_counter([v.selected_and_anderson_trend_test_class_counter for v in vs])
    selected_and_anderson_and_likelihood_counter = merge_counter(
        [v.selected_and_anderson_and_likelihood_ratio_trend_test_class_counter() for v in vs])

    total_of_selected_models = sum(selected_counter.values())
    l = sorted(enumerate(visualizer.non_stationary_trend_test), key=lambda e: selected_counter[e[1]])
    permutation = [i for i, v in l][::-1]

    select_list = get_ordered_list_from_counter(selected_counter, total_of_selected_models, visualizer, permutation)
    selected_and_anderson_list = get_ordered_list_from_counter(selected_and_anderson_counter, total_of_selected_models,
                                                               visualizer, permutation)
    selected_and_anderson_and_likelihood_list = get_ordered_list_from_counter(
        selected_and_anderson_and_likelihood_counter, total_of_selected_models, visualizer, permutation)

    labels = [get_trend_test_name(t) for t in visualizer.non_stationary_trend_test]
    labels = permute(labels, permutation)
    print(select_list, sum(select_list))

    nb_selected_models = sum(select_list)
    nb_selected_and_anderson_models = sum(selected_and_anderson_list)
    nb_selected_and_anderson_and_likelihood_models = sum(selected_and_anderson_and_likelihood_list)
    nb_selected_models_not_passing_any_test = nb_selected_models - nb_selected_and_anderson_models
    nb_selected_models_just_passing_anderson = nb_selected_and_anderson_models - nb_selected_and_anderson_and_likelihood_models

    # parameters
    width = 5
    size = 30
    legend_fontsize = 30
    labelsize = 15
    linewidth = 3
    x = [10 * (i + 1) for i in range(len(select_list))]
    for l, color, label, nb in zip([select_list, selected_and_anderson_list, selected_and_anderson_and_likelihood_list],
                               ['white', 'grey', 'black'],
                               ['Not passing any test', 'Just passing anderson test ',
                                'Passing both anderson and likelihood tests '],
                               [nb_selected_models_not_passing_any_test, nb_selected_models_just_passing_anderson, nb_selected_and_anderson_and_likelihood_models]):
        label += ' ({} \%)'.format(round(nb, 1))
        ax.bar(x, l, width=width, color=color, edgecolor='black', label=label, linewidth=linewidth)

    ax.legend(loc='upper right', prop={'size': size})
    ax.set_ylabel(' Frequency of selected model w.r.t all time series \n '
                  'i.e. for all massifs and altitudes (\%)', fontsize=legend_fontsize)
    ax.set_xlabel('Models', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()
    ax.set_xticklabels(labels)

    # Save plot
    visualizer.plot_name = 'Selection curves with significance level = {} '.format(AbstractGevTrendTest.SIGNIFICANCE_LEVEL)
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    plt.close()
