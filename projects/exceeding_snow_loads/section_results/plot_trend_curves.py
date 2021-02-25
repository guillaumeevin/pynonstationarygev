from typing import Dict
import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from projects.exceeding_snow_loads.utils import dpi_paper1_figure
from extreme_trend.trend_test.visualizers import \
    StudyVisualizerForNonStationaryTrends


def plot_trend_map(altitude_to_visualizer):
    # Compute common max value for the colorbar
    max_abs_changes_above_900 = [visualizer.max_abs_change
                                 for altitude, visualizer in altitude_to_visualizer.items()
                                 if altitude >= 900]
    max_abs_tdrl_above_900 = max(max_abs_changes_above_900) if len(max_abs_changes_above_900) > 0 else None

    for altitude, visualizer in altitude_to_visualizer.items():
        if 900 <= altitude <= 4200:
            add_color = (visualizer.study.altitude - 1800) % 1200 == 0
            visualizer.plot_trends(max_abs_tdrl_above_900, add_colorbar=add_color)
            # Plot 2700 also with a colorbar
            if altitude == 2700:
                visualizer.plot_trends(max_abs_tdrl_above_900, add_colorbar=True)
            if altitude == 1800:
                visualizer.plot_trends(max_abs_tdrl_above_900, add_colorbar=False)
        else:
            max_abs_tdrl_below_900 = max(altitude_to_visualizer[300].max_abs_change,
                                         altitude_to_visualizer[600].max_abs_change)
            visualizer.plot_trends(max_abs_tdrl_below_900, add_colorbar=altitude == 600)


def plot_trend_curves(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """
    Plot a single trend curves
    :return:
    """
    visualizer = list(altitude_to_visualizer.values())[0]

    ax = create_adjusted_axes(1, 1)
    ax_twinx = ax.twinx()
    ax_twiny = ax.twiny()

    trend_summary_values = list(zip(*[v.trend_summary_values() for v in altitude_to_visualizer.values()]))
    altitudes, percent_decrease, percent_decrease_signi, *mean_decreases = trend_summary_values

    # parameters
    width = 150
    legend_size = 30
    legend_fontsize = 35
    color = 'white'
    labelsize = 15
    linewidth = 3
    ax.bar(altitudes, percent_decrease, width=width, color=color, edgecolor='blue', label='decreasing trend',
           linewidth=linewidth)
    ax.bar(altitudes, percent_decrease_signi, width=width, color=color, edgecolor='black',
           label='significant decreasing trend',
           linewidth=linewidth)
    ax.legend(loc='upper left', prop={'size': legend_size})

    for ax_horizontal in [ax, ax_twiny]:
        if ax_horizontal == ax_twiny:
            ax_horizontal.plot(altitudes, [0 for _ in altitudes], linewidth=0)
        else:
            ax_horizontal.set_xlabel('Altitude', fontsize=legend_fontsize)
        ax_horizontal.set_xticks(altitudes)
        ax_horizontal.set_xlim([700, 5000])
        ax_horizontal.tick_params(labelsize=labelsize)

    # Set the number of massifs on the upper axis
    ax_twiny.set_xticklabels([v.study.nb_study_massif_names for v in altitude_to_visualizer.values()])
    ax_twiny.set_xlabel('Total number of massifs at each altitude (for the percentage)', fontsize=legend_fontsize)

    ax.set_ylabel('Massifs with decreasing trend (\%)', fontsize=legend_fontsize)
    max_percent = int(max(percent_decrease))
    n = 2 + (max_percent // 10)
    ax_ticks = [10 * i for i in range(n)]
    # upper_lim = max_percent + 3
    upper_lim = n + 5
    ax_lim = [0, upper_lim]
    for axis in [ax, ax_twinx]:
        axis.set_ylim(ax_lim)
        axis.set_yticks(ax_ticks)
        axis.tick_params(labelsize=labelsize)
    ax.yaxis.grid()

    label_curve = (visualizer.label).replace('change', 'decrease')
    ax_twinx.set_ylabel(label_curve.replace('', ''), fontsize=legend_fontsize)
    for region_name, mean_decrease in zip(AbstractExtendedStudy.region_names, mean_decreases):
        if len(mean_decreases) > 1:
            label = region_name
        else:
            label = 'Mean relative decrease'
        ax_twinx.plot(altitudes, mean_decrease, label=label, linewidth=linewidth, marker='o')
        ax_twinx.legend(loc='upper right', prop={'size': legend_size})

    # Save plot
    visualizer.plot_name = 'Trend curves'
    visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
    plt.close()
