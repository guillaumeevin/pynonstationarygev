from typing import Dict
import matplotlib.pyplot as plt
import numpy as np

from experiment.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR, EUROCODE_ALTITUDES
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import ci_method_to_color, \
    ci_method_to_label


def plot_uncertainty_histogram(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """ Plot one graph for each non-stationary context
    :return:
    """
    altitude_to_visualizer = {a: v for a, v in altitude_to_visualizer.items() if a in EUROCODE_ALTITUDES}
    visualizer = list(altitude_to_visualizer.values())[0]
    for non_stationary_context in visualizer.non_stationary_contexts:
        plot_histogram(altitude_to_visualizer, non_stationary_context)


def plot_histogram(altitude_to_visualizer, non_stationary_context):
    """
    Plot a single graph for potentially several confidence interval method
    :param altitude_to_visualizer:
    :param non_stationary_context:
    :return:
    """
    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    ax = plt.gca()
    altitudes = np.array(list(altitude_to_visualizer.keys()))
    bincenters = altitudes
    for j, ci_method in enumerate(visualizer.uncertainty_methods):
        if len(visualizer.uncertainty_methods) == 2:
            offset = -50 if j == 0 else 50
            bincenters = altitudes + offset
        width = 100
        plot_histogram_ci_method(visualizers, non_stationary_context, ci_method, ax, bincenters, width=width)
    ax.set_xticks(altitudes)
    ax.legend(loc='upper left')
    ax.set_ylabel('Massifs exceeding French standards (\%)')
    ax.set_xlabel('Altitude (m)')
    ax.set_ylim([0, 100])
    ax.set_yticks([10 * i for i in range(11)])
    visualizer.plot_name = 'Percentages of exceedance with non_stationary={}'.format(non_stationary_context)
    # visualizer.show = True
    visualizer.show_or_save_to_file(no_title=True, dpi=1000)
    ax.clear()


def plot_histogram_ci_method(visualizers, non_stationary_context, ci_method, ax, bincenters, width):
    three_percentages_of_excess = [v.three_percentages_of_excess(ci_method, non_stationary_context) for v in
                                   visualizers]
    epsilon = 0.5
    three_percentages_of_excess = [(a, b, c) if a == b else (max(epsilon, a), b, c) for (a, b, c) in three_percentages_of_excess]
    three_percentages_of_excess = [(a, b, c) if b == c else (a, b, min(100 - epsilon, c)) for (a, b, c) in three_percentages_of_excess]
    y = [d[1] for d in three_percentages_of_excess]
    yerr = np.array([[d[1] - d[0], d[2] - d[1]] for d in three_percentages_of_excess]).transpose()
    label = ci_method_to_label[ci_method]
    color = ci_method_to_color[ci_method]
    ax.bar(bincenters, y, width=width, color=color, yerr=yerr, label=label, ecolor='black', capsize=5)
