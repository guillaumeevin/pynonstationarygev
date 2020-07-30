from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np

from extreme_data.eurocode_data.utils import EUROCODE_ALTITUDES
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from projects.exceeding_snow_loads.utils import dpi_paper1_figure
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
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
    ylim = (0, 110)
    ticks = []
    for j, ci_method in enumerate(visualizer.uncertainty_methods):
        if len(visualizer.uncertainty_methods) == 2:
            offset = -50 if j == 0 else 50
            bincenters = altitudes + offset
            width = 100
        else:
            width = 200
        legend_size = 10
        # Plot histogram on the left axis
        plot_histogram_ci_method(visualizers, model_subset_for_uncertainty, ci_method, ax, bincenters, width, legend_size)

        # Plot percentages of return level excess on the right axis
        ylim, ticks = plot_percentage_of_excess(visualizers, model_subset_for_uncertainty, ci_method, ax.twinx(), fontsize_label, legend_size)


    ax.set_xticks(altitudes)
    ax.tick_params(labelsize=fontsize_label)
    if not (len(visualizer.uncertainty_methods) == 1
            and visualizer.uncertainty_methods[0] == ConfidenceIntervalMethodFromExtremes.ci_mle):
        ax.legend(loc='upper left', prop={'size': legend_size})
    # ax.set_ylabel('Massifs whose 50-year return level\n'
    #               'exceeds French standards (\%)', fontsize=fontsize_label)
    ax.set_ylabel('Massifs exceeding French standards (\%)', fontsize=fontsize_label)
    ax.set_xlabel('Altitude (m)', fontsize=fontsize_label)
    ax.set_ylim(ylim)
    ax.yaxis.grid()

    ax_twiny = ax.twiny()
    ax_twiny.plot(altitudes, [0 for _ in altitudes], linewidth=0)
    ax_twiny.tick_params(labelsize=fontsize_label)
    ax_twiny.set_xlim(ax.get_xlim())
    ax_twiny.set_xticks(altitudes)

    nb_massif_names = [len(v.intersection_of_massif_names_fitted) for v in altitude_to_visualizer.values()]
    ax_twiny.set_xticklabels(nb_massif_names)
    ax_twiny.set_xlabel('Number of massifs at each altitude (for the percentage and the mean)', fontsize=fontsize_label)


    ax.set_yticks(ticks)
    visualizer.plot_name = 'Percentages of exceedance with {}'.format(
        get_display_name_from_object_type(model_subset_for_uncertainty))
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure, tight_layout=True)
    ax.clear()
    ax_twiny.clear()
    plt.close()


def plot_histogram_ci_method(visualizers, model_subset_for_uncertainty, ci_method, ax, bincenters, width, legend_size=10):
    three_percentages_of_excess = [v.excess_metrics(ci_method, model_subset_for_uncertainty)[0] for v in
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
    ecolor = 'black'
    label_name = 'Percentage of massifs exceeding'
    # ax.bar(bincenters, y, width=width, color=color, yerr=yerr, label=label_name, ecolor=ecolor, capsize=5)
    ax.bar(bincenters, y, width=width, color=color, label=label_name)
    # Just to add something in the legend
    label_confidence_interval = get_label_confidence_interval(label_name)
    ax.errorbar(bincenters, y, yerr=yerr, label=label_confidence_interval,
                fmt='none', color=ecolor, capsize=5)

    ax.legend(loc='upper left', prop={'size': legend_size})


def plot_percentage_of_excess(visualizers, model_subset_for_uncertainty, ci_method, ax, fontsize_label, legend_size=10) -> Tuple[int, int]:
    l = [v.excess_metrics(ci_method, model_subset_for_uncertainty)[2] for v in visualizers]
    lower_bound, mean, upper_bound = list(zip(*l))
    other_mean = [e[1] for e in l]
    altitudes = [v.altitude for v in visualizers]
    # Display parameters
    color = 'blue'
    alpha = 0.2
    full_label_name = 'Mean relative difference between\n' \
                      '50-year return levels and French standards (\%)'

    label_name = 'Mean relative difference'
    ax.plot(altitudes, mean, linestyle='--', marker='o', color=color,
            label=label_name)

    label_confidence_interval = get_label_confidence_interval(label_name)
    ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha,
                    label=label_confidence_interval)

    ax.tick_params(labelsize=fontsize_label)
    ax.legend(loc='lower right', prop={'size': legend_size})
    ax.set_ylabel(full_label_name, fontsize=fontsize_label)

    ylim = (-85, 110)
    ax.set_ylim(ylim)

    nb_ticks =  1 + (ylim[1] - ylim[0]) // 20
    ticks = [100 - 20 * i for i in range(nb_ticks)][::-1]
    ax.set_yticks(ticks)

    return ylim, ticks


def get_label_confidence_interval(label_name):
    confidence_interval_str = '\n{}'.format(AbstractExtractEurocodeReturnLevel.percentage_confidence_interval)
    confidence_interval_str += ' \% confidence interval'
    label_confidence_interval = label_name + confidence_interval_str
    return label_confidence_interval

