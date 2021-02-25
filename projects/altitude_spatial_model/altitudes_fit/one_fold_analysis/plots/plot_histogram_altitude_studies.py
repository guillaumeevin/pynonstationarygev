from typing import List

import numpy as np

import matplotlib

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.lines import Line2D

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plots.compute_histogram_change_in_total_snowfall import \
    compute_changes_in_total_snowfall


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


def plot_histogram_all_trends_against_altitudes(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], with_significance=True):
    visualizer = visualizer_list[0]

    all_trends = [v.all_trends(massif_names, with_significance=with_significance) for v in visualizer_list]
    nb_massifs, *all_l = zip(*all_trends)

    plt.close()
    ax = plt.gca()
    width = 6
    size = 10
    legend_fontsize = 15
    labelsize = 10
    linewidth = 3
    x = np.array([3 * width * (i + 1) for i in range(len(nb_massifs))])

    colors = ['blue', 'darkblue', 'red', 'darkred']
    # colors = ['red', 'darkred', 'limegreen', 'darkgreen']
    labels = []
    for suffix in ['decrease', 'increase']:
        for prefix in ['Non significant', 'Significant']:
            labels.append('{} {}'.format(prefix, suffix))
    for l, color, label in zip(all_l, colors, labels):
        shift = 0.6 * width
        is_a_decrease_plot = colors.index(color) in [0, 1]
        x_shifted = x - shift if is_a_decrease_plot else x + shift
        if with_significance or (not label.startswith("S")):
            ax.bar(x_shifted, l, width=width, color=color, edgecolor=color, label=label,
                   linewidth=linewidth, align='center')
    ax.legend(loc='upper left', prop={'size': size})
    ax.set_ylabel('Percentage of massifs (\%) ', fontsize=legend_fontsize)
    ax.set_xlabel('Elevation range', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()
    _, ylim_max = ax.get_ylim()
    ax.set_ylim([0, max(ylim_max, 79)])
    ax.set_xticklabels([v.altitude_group.formula_upper for v in visualizer_list])

    plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x)

    visualizer.plot_name = 'All trends'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()

def plot_shoe_plot_ratio_interval_size_against_altitude(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels]):
    visualizer = visualizer_list[0]

    ratio_groups = []
    for v in visualizer_list:
        ratio_groups.extend(v.ratio_groups())
    print(len(ratio_groups))
    print(ratio_groups)


    nb_massifs = [len(l) for l in ratio_groups]

    plt.close()
    ax = plt.gca()
    width = 5
    size = 8
    legend_fontsize = 10
    labelsize = 10
    linewidth = 3

    x = np.array([2 * width * (i + 1) for i in range(len(ratio_groups))])
    ax.boxplot(ratio_groups, positions=x, widths=width, patch_artist=True, showmeans=True)

    ax.legend(prop={'size': 8})

    ylabel = "Ratio for the size of {}\% confidence intervals, i.e. size for the\n" \
    " elevational-temporal model divided by the size for the pointwise model".format(AbstractExtractEurocodeReturnLevel.percentage_confidence_interval)
    ax.set_ylabel(ylabel,
                  fontsize=legend_fontsize)
    ax.set_xlabel('Elevation (m)', fontsize=legend_fontsize + 5)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()

    altitudes = []
    for v in visualizer_list:
        altitudes.extend(v.studies.altitudes)
    ax.set_xticklabels([str(a) for a in altitudes])

    shift = 2 * width
    ax.set_xlim((min(x) - shift, max(x) + shift))

    # I could display the number of massif used to build each box plot.
    plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x, add_for_percentage=False)

    visualizer.plot_name = 'All ' + "uncertainty size comparison for bootstrap size {}".format(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_shoe_plot_changes_against_altitude(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels],
                                            relative=False, with_significance=True):
    visualizer = visualizer_list[0]

    all_changes = [v.all_changes(massif_names, relative=relative, with_significance=with_significance) for v in visualizer_list]
    all_changes = list(zip(*all_changes))
    labels = ['All massifs', 'Massifs with a selected model\n'
                             'temporally non-stationary',
              'Massifs with a selected model\n'
              'temporally non-stationary and significant']
    colors = ['darkgreen', 'forestgreen', 'limegreen']
    if not with_significance:
        labels = labels[:-1]
        colors = colors[:-1]
    nb_massifs = [len(v.get_valid_names(massif_names)) for v in visualizer_list]

    plt.close()
    ax = plt.gca()
    width = 5
    size = 8
    legend_fontsize = 10
    labelsize = 10

    x = np.array([4 * width * (i + 1) for i in range(len(nb_massifs))])
    for j, (changes, label, color) in enumerate(list(zip(all_changes, labels, colors)), -1):
        positions = x + j * width
        bplot = ax.boxplot(list(changes), positions=positions, widths=width, patch_artist=True, showmeans=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    loc = 'upper right' if relative else 'upper left'
    ax.legend(custom_lines, labels, prop={'size': 12}, loc=loc)

    start = 'Relative changes' if relative else 'Changes'
    unit = '\%' if relative else visualizer.study.variable_unit
    ax.set_ylabel('{} of {}-year return levels {} ({})'.format(start, OneFoldFit.return_period,
                                                               visualizer.first_one_fold_fit.between_covariate_str,
                                                                                  unit),
                  fontsize=legend_fontsize)
    ax.set_xlabel('Elevation (m)', fontsize=legend_fontsize + 5)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()

    altitudes = [v.altitude_group.reference_altitude for v in visualizer_list]
    ax.set_xticklabels([str(a) for a in altitudes])

    shift = 2 * width
    ax.set_xlim((min(x) - shift, max(x) + shift))

    if not isinstance(visualizer.study, AbstractAdamontStudy):
        upper_limit_for_legend = 50 if relative else 0
        lim_down, lim_up = ax.get_ylim()
        ax.set_ylim(lim_down, lim_up + upper_limit_for_legend)

    # Plot a zero horizontal line
    lim_left, lim_right = ax.get_xlim()
    ax.hlines(0, xmin=lim_left, xmax=lim_right, linestyles='dashed')

    # I could display the number of massif used to build each box plot.
    # plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x)

    visualizer.plot_name = 'All ' + start
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_shoe_plot_changes_against_altitude_for_maxima_and_total(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels],
                                            relative=False):
    visualizer = visualizer_list[0]

    all_changes_total= compute_changes_in_total_snowfall(visualizer_list,
                                                         relative)

    all_changes_extreme = [v.all_changes(massif_names, relative=relative) for v in visualizer_list]
    all_changes_extreme = list(zip(*all_changes_extreme))[:1][0]

    all_changes = [all_changes_extreme,  all_changes_total]
    labels = ['{}-year return levels'.format(OneFoldFit.return_period), 'Total snowfall']
    colors = ['darkgreen', 'royalblue']
    nb_massifs = [len(v.get_valid_names(massif_names)) for v in visualizer_list]

    plt.close()
    ax = plt.gca()
    width = 5
    size = 8
    legend_fontsize = 15
    labelsize = 10
    linewidth = 3

    x = np.array([3 * width * (i + 1) for i in range(len(nb_massifs))])
    for j, (changes, label, color) in enumerate(list(zip(all_changes, labels, colors))):
        print(len(changes), changes, label)
        positions = x + (2 * j - 1) * 0.5 * width
        bplot = ax.boxplot(changes, positions=positions, widths=width, patch_artist=True, showmeans=True)
        for patch in bplot['boxes']:
            patch.set_facecolor(color)

    custom_lines = [Line2D([0], [0], color=color, lw=4) for color in colors]
    loc = 'lower right' if relative else 'upper left'
    ax.legend(custom_lines, labels, loc=loc)

    start = 'Relative changes' if relative else 'Changes'
    unit = '\%' if relative else visualizer.study.variable_unit
    ax.set_ylabel('{} between 1959 and 2019 ({})'.format(start, unit),
                  fontsize=legend_fontsize)
    ax.set_xlabel('Elevation', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(x)
    ax.yaxis.grid()

    altitudes = [v.altitude_group.reference_altitude for v in visualizer_list]
    ax.set_xticklabels([str(a) for a in altitudes])

    shift = 2 * width
    ax.set_xlim((min(x) - shift, max(x) + shift))

    # I could display the number of massif used to build each box plot.
    # plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x)

    visualizer.plot_name = 'All ' + start + ' and total '
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_nb_massif_on_upper_axis(ax, labelsize, legend_fontsize, nb_massifs, x, add_for_percentage=True):
    # Plot number of massifs on the upper axis
    ax_twiny = ax.twiny()
    ax_twiny.plot(x, [0 for _ in x], linewidth=0)
    ax_twiny.set_xticks(x)
    ax_twiny.tick_params(labelsize=labelsize)
    ax_twiny.set_xticklabels(nb_massifs)
    ax_twiny.set_xlim(ax.get_xlim())
    xlabel = 'Total number of massifs at each range'
    if add_for_percentage:
        xlabel += ' (for the percentage)'
    ax_twiny.set_xlabel(xlabel, fontsize=legend_fontsize)
