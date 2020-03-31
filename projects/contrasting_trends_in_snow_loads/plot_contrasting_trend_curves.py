from typing import Dict
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from projects.exceeding_snow_loads.utils import dpi_paper1_figure
from extreme_trend.visualizers.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


def plot_contrasting_trend_curves_massif(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                  all_regions=False):
    """
    Plot a single trend curves
    :return:
    """
    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    altitudes = list(altitude_to_visualizer.keys())

    ax = create_adjusted_axes(1, 1)
    # ax_twinx = ax.twinx()
    ax_twinx = ax
    ax_twiny = ax.twiny()

    # parameters
    width = 150
    size = 20
    legend_fontsize = 20
    color = 'white'
    labelsize = 15
    linewidth = 3

    for ax_horizontal in [ax, ax_twiny]:
        if ax_horizontal == ax_twiny:
            ax_horizontal.plot(altitudes, [0 for _ in altitudes], linewidth=0)
        else:
            ax_horizontal.set_xlabel('Altitude', fontsize=legend_fontsize)
        ax_horizontal.set_xticks(altitudes)
        # ax_horizontal.set_xlim([700, 5000])
        ax_horizontal.tick_params(labelsize=labelsize)

    # Set the number of massifs on the upper axis
    ax_twiny.set_xticklabels([v.study.nb_study_massif_names for v in altitude_to_visualizer.values()])
    ax_twiny.set_xlabel('Total number of massifs at each altitude (for the percentage)', fontsize=legend_fontsize)

    ax_twinx.yaxis.grid()

    ax_twinx.set_ylabel(visualizer.label, fontsize=legend_fontsize)
    for j, massif_name in enumerate(visualizer.study.study_massif_names):
        massif_visualizers = [v for v in visualizers if massif_name in v.massif_name_to_change_value]
        changes = [v.massif_name_to_relative_change_value[massif_name] for v in massif_visualizers]
        massif_altitudes = [v.study.altitude for v in massif_visualizers]
        label = massif_name.replace('-', '').replace('_', '')
        if j < 10:
            linestyle = 'solid'
        elif j < 20:
            linestyle = 'dashed'
        else:
            linestyle = 'dotted'
        ax_twinx.plot(massif_altitudes, changes, label=label, linewidth=linewidth, marker='o', linestyle=linestyle)
        ax_twinx.legend(loc='upper right', prop={'size': 5})

    ax.axhline(y=0, color='k')

    # Save plot
    visualizer.plot_name = 'Trend curves for' + visualizer.study.variable_name
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure, folder_for_variable=False)
    plt.close()


def plot_contrasting_trend_curves(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                  all_regions=False):
    """
    Plot a single trend curves
    :return:
    """
    visualizer = list(altitude_to_visualizer.values())[0]

    ax = create_adjusted_axes(1, 1)
    # ax_twinx = ax.twinx()
    ax_twinx = ax
    ax_twiny = ax.twiny()

    trend_summary_values = list(zip(*[v.trend_summary_contrasting_values(regions=all_regions) for v in altitude_to_visualizer.values()]))
    altitudes, *mean_changes = trend_summary_values

    # parameters
    width = 150
    size = 20
    legend_fontsize = 20
    color = 'white'
    labelsize = 15
    linewidth = 3
    # ax.bar(altitudes, percent_decrease, width=width, color=color, edgecolor='blue', label='decreasing trend',
    #        linewidth=linewidth)
    # ax.bar(altitudes, percent_decrease_signi, width=width, color=color, edgecolor='black',
    #        label='significative decreasing trend',
    #        linewidth=linewidth)
    # ax.legend(loc='upper left', prop={'size': size})

    for ax_horizontal in [ax, ax_twiny]:
        if ax_horizontal == ax_twiny:
            ax_horizontal.plot(altitudes, [0 for _ in altitudes], linewidth=0)
        else:
            ax_horizontal.set_xlabel('Altitude', fontsize=legend_fontsize)
        ax_horizontal.set_xticks(altitudes)
        # ax_horizontal.set_xlim([700, 5000])
        ax_horizontal.tick_params(labelsize=labelsize)

    # Set the number of massifs on the upper axis
    ax_twiny.set_xticklabels([v.study.nb_study_massif_names for v in altitude_to_visualizer.values()])
    ax_twiny.set_xlabel('Total number of massifs at each altitude (for the percentage)', fontsize=legend_fontsize)

    # ax.set_ylabel('Massifs with decreasing trend (\%)', fontsize=legend_fontsize)
    # max_percent = int(max(percent_decrease))
    # n = 2 + (max_percent // 10)
    # ax_ticks = [10 * i for i in range(n)]
    # # upper_lim = max_percent + 3
    # upper_lim = n + 5
    # ax_lim = [0, upper_lim]
    # for axis in [ax, ax_twinx]:
    #     axis.set_ylim(ax_lim)
    #     axis.set_yticks(ax_ticks)
    #     axis.tick_params(labelsize=labelsize)
    ax_twinx.yaxis.grid()

    ax_twinx.set_ylabel(visualizer.label, fontsize=legend_fontsize)
    for j, (region_name, mean_change) in enumerate(zip(AbstractExtendedStudy.region_names, mean_changes)):
        if len(mean_changes) > 2:
            label = region_name
        elif len(mean_changes) == 2:
            label = 'North' if j == 0 else 'South'
        else:
            label = 'Mean relative change'
        ax_twinx.plot(altitudes, mean_change, label=label, linewidth=linewidth, marker='o')
        ax_twinx.legend(loc='upper right', prop={'size': size})

    ax.axhline(y=0, color='k')

    # Save plot
    visualizer.plot_name = 'Trend curves for' + visualizer.study.variable_name
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure, folder_for_variable=False)
    plt.close()
