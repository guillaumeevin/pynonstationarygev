from typing import List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


def drawPieMarker(ax, xs, ys, probabilities, sizes, colors):
    assert sum(probabilities) == 1, 'sum of ratios needs to be = 1'

    markers = []
    previous = 0
    # calculate the points of the pie pieces
    for color, probability in zip(colors, probabilities):
        this = 2 * np.pi * probability + previous
        x = [0] + np.cos(np.linspace(previous, this, 20)).tolist() + [0]
        y = [0] + np.sin(np.linspace(previous, this, 20)).tolist() + [0]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max() ** 2 * np.array(sizes), 'facecolor': color})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)


def plot_pychart_scatter_plot(visualizer_list, all_massif_names, covariates, with_return_level):
    ax = plt.gca()
    sizes = [1500 for _ in range(2)]
    altitudes = []
    list_nb_valid_massifs = []
    for visualizer in visualizer_list:
        altitude = visualizer.study.altitude
        altitudes.append(altitude)
        for covariate in covariates:
            OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
            nb_valid_massifs, *all_l = visualizer.all_trends(all_massif_names.copy(), with_significance=False,
                                              with_return_level=with_return_level)
            print("scatter plot", altitude, with_return_level, len(all_massif_names), nb_valid_massifs)
            if covariate == covariates[0]:
                list_nb_valid_massifs.append(nb_valid_massifs)
            decreasing_percentage = all_l[0] / 100
            if decreasing_percentage == 1:
                probabilities = [0.25 for _ in range(4)]
                colors = ['blue' for _ in range(4)]
            elif decreasing_percentage == 0:
                probabilities = [0.25 for _ in range(4)]
                colors = ['red' for _ in range(4)]
            else:
                probabilities = [decreasing_percentage / 2, decreasing_percentage / 2,
                                 (1 - decreasing_percentage) / 2, (1 - decreasing_percentage) / 2]
                colors = ['blue', 'blue', 'red', 'red']
            drawPieMarker(ax, covariate, altitude, probabilities, sizes, colors)

            # add text with the percentage
            epsilon = 40
            shift = -20
            fontsize = 6
            if decreasing_percentage in [0, 1]:
                ax.text(covariate - 0.045,
                        shift + altitude,
                        "100\%",
                        fontsize=fontsize, weight='bold')
            else:
                for j, probability in enumerate([1-decreasing_percentage, decreasing_percentage]):
                    ax.text(covariate + 0.055,
                            shift + altitude - epsilon + 2 * j * epsilon,
                            str(int(100 * probability)) + "\%",
                            fontsize=fontsize, weight='bold')

    legend_fontsize = 10
    labelsize = 10
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(covariates)
    ax.set_yticks(altitudes)
    mi, ma = ax.get_ylim()
    border = 200
    ax.set_ylim((mi - border, ma + border + 450))
    ax.set_yticklabels(["{} m".format(v.study.altitude) for v in visualizer_list])
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates])

    ax2 = ax.twinx()
    ax2.set_ylabel('Number of massifs')
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([str(nb) for nb in list_nb_valid_massifs])
    # Build legend
    custom_lines = [Line2D([0], [0], color=color, lw=6) for color in ['blue', 'red']]
    # custom_lines = [Patch(facecolor=color, edgecolor=color) for color in ['blue', 'red']]
    metric = '{}-year return levels'.format(OneFoldFit.return_period) if with_return_level else 'mean annual maxima'
    labels = ['\% of massifs with {} in {} between +T and +1 degrees'.format(s, metric) for s in ['a decrease', 'an increase']]
    ax.legend(custom_lines, labels, prop={'size': 8.5}, loc='upper left' )

    label = "return level" if with_return_level else "mean"
    visualizer = visualizer_list[0]
    visualizer.plot_name = 'Piece charts for {}'.format(label)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_relative_change_at_massif_level(visualizer_list, massif_name, with_return_level,
                                         with_significance, relative_change, return_period
                                         ):
    default_return_period = OneFoldFit.return_period
    if with_return_level:
        OneFoldFit.return_period = return_period
    colors = ['k', 'forestgreen', 'limegreen', 'yellowgreen', 'greenyellow']
    covariates = np.linspace(1, 4, num=100)
    covariates_to_show = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    ax = plt.gca()
    for color, visualizer in zip(colors, visualizer_list[::-1]):
        altitude = visualizer.study.altitude
        label = '{} m'.format(altitude)
        visualizer: AltitudesStudiesVisualizerForNonStationaryModels
        if massif_name in visualizer.massif_name_to_one_fold_fit:
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
            order = None if with_return_level else 1
            f = one_fold_fit.relative_changes_of_moment if relative_change else one_fold_fit.changes_of_moment
            y = [f([altitude], order, 1, c)[0] for c in covariates]
            ax.plot(covariates, y, label=label, color=color, linewidth=3)

    legend_fontsize = 10
    labelsize = 10
    massif_name_str = massif_name.replace('_', '-')
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    change_str = 'Relative change' if relative_change else 'Change'
    unit = '\%' if relative_change else visualizer.study.variable_unit
    metric = "{}-year return level".format(OneFoldFit.return_period) if with_return_level else "mean annual maxima"
    ylabel = '{} in {} w.r.t +1 degree\nof global warming for the {} massif ({})'.format(change_str, metric, massif_name_str, unit)
    ax.set_ylabel(ylabel, fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(covariates_to_show)
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show])
    ax.set_xlim((1, 4))

    ax.legend(prop={'size': 12}, loc='lower left')

    visualizer = visualizer_list[0]
    visualizer.plot_name = '{}/{} of {} for {}'.format(massif_name_str, change_str, metric, massif_name_str)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    OneFoldFit.return_period = default_return_period

    plt.close()

def plot_relative_change_at_massif_level_sensitivity_to_frequency(visualizer : AltitudesStudiesVisualizerForNonStationaryModels, massif_name,
                                         with_significance, relative_change,
                                                                  return_periods
                                         ):
    default_return_period = OneFoldFit.return_period
    altitude = visualizer.study.altitude
    colors = ['k', 'darkmagenta', 'darkviolet', 'blueviolet', 'mediumpurple', 'plum']
    covariates = np.linspace(1, 4, num=100)
    covariates_to_show = [1, 1.5, 2, 2.5, 3, 3.5, 4]
    ax = plt.gca()
    return_periods = return_periods[::-1]
    for return_period, color in zip(return_periods, colors):
        OneFoldFit.return_period = return_period
        label = '{}-year return level'.format(return_period)
        if massif_name in visualizer.massif_name_to_one_fold_fit:
            one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
            f = one_fold_fit.relative_changes_of_moment if relative_change else one_fold_fit.changes_of_moment
            y = [f([altitude], None, 1, c)[0] for c in covariates]
            ax.plot(covariates, y, label=label, color=color, linewidth=3)

    legend_fontsize = 10
    labelsize = 10
    massif_name_str = massif_name.replace('_', '-')
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    change_str = 'Relative change' if relative_change else 'Change'
    unit = '\%' if relative_change else visualizer.study.variable_unit
    metric = "return levels".format(OneFoldFit.return_period)
    ylabel = '{} in {} w.r.t +1 degree of global warming\nfor the {} massif at {} m ({})'.format(change_str, metric,
                                                                                                 massif_name_str,
                                                                                                 altitude,
                                                                                                 unit)
    ax.set_ylabel(ylabel, fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks(covariates_to_show)
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show])
    ax.set_xlim((1, 4))
    ax.legend(prop={'size': 10}, loc='upper right')


    visualizer.plot_name = '{}/{} of {} for {}'.format(massif_name_str, change_str, metric, massif_name_str)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    OneFoldFit.return_period = default_return_period

    plt.close()
