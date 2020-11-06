import math
from typing import List

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit


def plot_histogram_all_models_against_altitudes(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels]):
    visualizer = visualizer_list[0]
    model_names = visualizer.model_names
    model_name_to_percentages = {model_name: [] for model_name in model_names}
    model_name_to_percentages_significant = {model_name: [] for model_name in model_names}
    for v in visualizer_list:
        model_name_to_percentages_for_v = v.model_name_to_percentages(massif_names, only_significant=False)
        model_name_to_significant_percentages_for_v = v.model_name_to_percentages(massif_names, only_significant=True)
        for model_name in model_names:
            model_name_to_percentages[model_name].append(model_name_to_percentages_for_v[model_name])
            model_name_to_percentages_significant[model_name].append(
                model_name_to_significant_percentages_for_v[model_name])
    # Sort model based on their mean percentage.
    model_name_to_mean_percentage = {m: np.mean(a) for m, a in model_name_to_percentages.items()}
    model_name_to_mean_percentage_significant = {m: np.mean(a) for m, a in
                                                 model_name_to_percentages_significant.items()}
    sorted_model_names = sorted(model_names, key=lambda m: model_name_to_mean_percentage[m], reverse=True)
    for model_name in sorted_model_names:
        print(model_name_to_mean_percentage[model_name], model_name_to_mean_percentage_significant[model_name],
              model_name)

    # Plot part
    ax = plt.gca()
    width = 5
    size = 8
    legend_fontsize = 10
    labelsize = 10
    linewidth = 1
    tick_list = np.array([((len(visualizer_list) + 2) * i + (1 + len(visualizer_list) / 2)) * width
                          for i in range(len(sorted_model_names))])
    for tick_middle, model_name in zip(tick_list, sorted_model_names):
        x_shifted = [tick_middle + width * shift / 2 for shift in range(-3, 5, 2)]
        percentages = model_name_to_percentages[model_name]
        percentages_significant = model_name_to_percentages_significant[model_name]
        colors = ['white', 'yellow', 'orange', 'red']
        labels = ['{} m - {} m (\% out of {} massifs)'.format(1000 * i, 1000 * (i + 1),
                                                              len(v.get_valid_names(massif_names))) for i, v in
                  enumerate(visualizer_list)]
        for x, color, percentage, label, percentage_significant in zip(x_shifted, colors, percentages, labels,
                                                                       percentages_significant):
            ax.bar([x], [percentage], width=width, label=label,
                   linewidth=2 * linewidth, edgecolor='black', color=color)
            heights = list(range(0, math.ceil(percentage_significant), 1))[::-1]
            for height in heights:
                ax.bar([x], [height], width=width, linewidth=linewidth, edgecolor='black', color=color)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:len(visualizer_list)], labels[:len(visualizer_list)], prop={'size': size})
    ax.set_xticklabels(sorted_model_names)
    ax.set_xticks(tick_list)
    ax.set_ylabel('Percentage of massifs (\%) ', fontsize=legend_fontsize)
    ax.set_xlabel('Models', fontsize=legend_fontsize)
    ax.set_ylim(bottom=0)
    ax.yaxis.grid()
    ax.tick_params(axis='both', which='major', labelsize=labelsize)

    visualizer.plot_name = 'All models'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()


def plot_histogram_all_trends_against_altitudes(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels]):
    visualizer = visualizer_list[0]

    all_trends = [v.all_trends(massif_names) for v in visualizer_list]
    nb_massifs, *all_l = zip(*all_trends)

    plt.close()
    ax = plt.gca()
    width = 5
    size = 10
    legend_fontsize = 15
    labelsize = 10
    linewidth = 3
    x = np.array([3 * width * (i + 1) for i in range(len(nb_massifs))])

    colors = ['blue', 'darkblue', 'red', 'darkred']
    labels = []
    for suffix in ['Decrease', 'Increase']:
        for prefix in ['Non significant', 'Significant']:
            labels.append('{} {}'.format(prefix, suffix))
    for l, color, label in zip(all_l, colors, labels):
        x_shifted = x - width / 2 if 'blue' in color else x + width / 2
        ax.bar(x_shifted, l, width=width, color=color, edgecolor=color, label=label,
               linewidth=linewidth)
    ax.legend(loc='upper left', prop={'size': size})
    ax.set_ylabel('Percentage of massifs (\%) ', fontsize=legend_fontsize)
    ax.set_xlabel('Elevation range', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()
    ax.set_ylim(bottom=0)
    ax.set_xticklabels([v.altitude_group.formula_upper for v in visualizer_list])

    plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x)

    visualizer.plot_name = 'All trends'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_shoe_plot_changes_against_altitude(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels],
                                            relative=False):
    visualizer = visualizer_list[0]

    all_changes = [v.all_changes(massif_names, relative=relative) for v in visualizer_list]
    all_changes = list(zip(*all_changes))
    labels = ['All selected models', 'Temporally non-stationary models',
              'Temporally non-stationary models which are significant']
    colors = ['darkgreen', 'forestgreen', 'limegreen']
    nb_massifs = [len(v.get_valid_names(massif_names)) for v in visualizer_list]

    plt.close()
    ax = plt.gca()
    width = 5
    size = 8
    legend_fontsize = 10
    labelsize = 10
    linewidth = 3

    x = np.array([4 * width * (i + 1) for i in range(len(nb_massifs))])
    for j, (changes, label, color) in enumerate(list(zip(all_changes, labels, colors)), -1):
        positions = x + j * width
        bplot = ax.boxplot(list(changes), positions=positions, widths=width, patch_artist=True, showmeans=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    loc = 'lower right' if relative else 'upper left'
    ax.legend(custom_lines, labels, loc=loc)

    start = 'Relative changes' if relative else 'Changes'
    unit = '\%' if relative else visualizer.study.variable_unit
    ax.set_ylabel('{} of {}-year return levels between 1969 and 2019 ({})'.format(start, OneFoldFit.return_period,
                                                                                  unit),
                  fontsize=legend_fontsize)
    ax.set_xlabel('Elevation', fontsize=legend_fontsize + 5)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()

    altitudes = [v.altitude_group.reference_altitude for v in visualizer_list]
    ax.set_xticklabels([str(a) for a in altitudes])

    shift = 2 * width
    ax.set_xlim((min(x) - shift, max(x) + shift))

    # I could display the number of massif used to build each box plot.
    # plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x)

    visualizer.plot_name = 'All ' + start
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x):
    # Plot number of massifs on the upper axis
    ax_twiny = ax.twiny()
    ax_twiny.plot(x, [0 for _ in x], linewidth=0)
    ax_twiny.set_xticks(x)
    ax_twiny.tick_params(labelsize=labelsize)
    ax_twiny.set_xticklabels(nb_massifs)
    ax_twiny.set_xlim(ax.get_xlim())
    ax_twiny.set_xlabel('Total number of massifs at each range (for the percentage)', fontsize=legend_fontsize)
