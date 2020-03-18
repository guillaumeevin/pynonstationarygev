from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from experiment.eurocode_data.utils import EUROCODE_ALTITUDES
from projects.exceeding_snow_loads.utils import dpi_paper1_figure
from extreme_trend_test.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import ci_method_to_color, \
    ci_method_to_label, ConfidenceIntervalMethodFromExtremes
from root_utils import get_display_name_from_object_type


def plot_uncertainty_histogram(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """ Plot one graph for each non-stationary context
    :return:
    """
    altitude_to_visualizer = {a: v for a, v in altitude_to_visualizer.items() if a in EUROCODE_ALTITUDES}
    visualizer = list(altitude_to_visualizer.values())[0]
    for model_subset_for_uncertainty in visualizer.model_subsets_for_uncertainty:
        plot_histogram(altitude_to_visualizer, model_subset_for_uncertainty)


def plot_histogram(altitude_to_visualizer, model_subset_for_uncertainty):
    """
    Plot a single graph for potentially several confidence interval method
    :param altitude_to_visualizer:
    :param model_subset_for_uncertainty:
    :return:
    """
    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    ax = plt.gca()
    altitudes = np.array(list(altitude_to_visualizer.keys()))
    bincenters = altitudes

    fontsize_label = 15
    legend_size = 15
    # Plot histogram
    for j, ci_method in enumerate(visualizer.uncertainty_methods):
        if len(visualizer.uncertainty_methods) == 2:
            offset = -50 if j == 0 else 50
            bincenters = altitudes + offset
            width = 100
        else:
            width = 200
        plot_histogram_ci_method(visualizers, model_subset_for_uncertainty, ci_method, ax, bincenters, width=width)

    ax.set_xticks(altitudes)
    ax.tick_params(labelsize=fontsize_label)
    if not (len(visualizer.uncertainty_methods) == 1
            and visualizer.uncertainty_methods[0] == ConfidenceIntervalMethodFromExtremes.ci_mle):
        ax.legend(loc='upper left', prop={'size': legend_size})
    # ax.set_ylabel('Massifs whose 50-year return level\n'
    #               'exceeds French standards (\%)', fontsize=fontsize_label)
    ax.set_ylabel('Massifs exceeding French standards (\%)', fontsize=fontsize_label)
    ax.set_xlabel('Altitude (m)', fontsize=fontsize_label)
    ax.set_ylim([0, 100])
    ax.yaxis.grid()

    ax_twiny = ax.twiny()
    ax_twiny.plot(altitudes, [0 for _ in altitudes], linewidth=0)
    ax_twiny.tick_params(labelsize=fontsize_label)
    ax_twiny.set_xlim(ax.get_xlim())
    ax_twiny.set_xticks(altitudes)
    nb_massif_names = [len(v.massif_names_fitted) for v in altitude_to_visualizer.values()]
    ax_twiny.set_xticklabels(nb_massif_names)
    ax_twiny.set_xlabel('Total number of massifs at each altitude (for the percentage)', fontsize=fontsize_label)

    ax.set_yticks([10 * i for i in range(11)])
    visualizer.plot_name = 'Percentages of exceedance with {}'.format(
        get_display_name_from_object_type(model_subset_for_uncertainty))
    # visualizer.show = True
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    ax.clear()
    ax_twiny.clear()
    plt.close()


def plot_histogram_ci_method(visualizers, model_subset_for_uncertainty, ci_method, ax, bincenters, width):
    three_percentages_of_excess = [v.excess_metrics(ci_method, model_subset_for_uncertainty)[:3] for v in
                                   visualizers]
    epsilon = 0.5
    three_percentages_of_excess = [(a, b, c) if a == b else (max(epsilon, a), b, c) for (a, b, c) in
                                   three_percentages_of_excess]
    three_percentages_of_excess = [(a, b, c) if b == c else (a, b, min(100 - epsilon, c)) for (a, b, c) in
                                   three_percentages_of_excess]
    y = [d[1] for d in three_percentages_of_excess]
    yerr = np.array([[d[1] - d[0], d[2] - d[1]] for d in three_percentages_of_excess]).transpose()
    label = ci_method_to_label[ci_method]
    color = ci_method_to_color[ci_method]
    ax.bar(bincenters, y, width=width, color=color, yerr=yerr, label=label, ecolor='black', capsize=5)
