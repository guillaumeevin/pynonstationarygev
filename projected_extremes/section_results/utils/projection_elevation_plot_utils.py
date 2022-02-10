import math
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_averaged_global_mean_temp, \
    get_closest_year
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_shifted_map, \
    get_colors, create_colorbase_axis, ticks_values_and_labels_for_percentages, get_upper_two_third_colormap, \
    get_lower_half_colormap, get_upper_half_colormap, ticks_values_and_labels_for_half_value
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from extreme_trend.one_fold_fit.plots.plot_histogram_altitude_studies import plot_nb_massif_on_upper_axis


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


def drawPieMarker(ax, xs, ys, probabilities, sizes, edges_colors, pie_color, color_to_linestyle, linewidth):
    assert sum(probabilities) == 1, 'sum of ratios needs to be = 1'

    markers = []
    previous = 0
    standard_number_lines = 26

    # calculate the points of the pie pieces
    for j, (color, probability) in enumerate(zip(edges_colors, probabilities)):
        this = 2 * np.pi * probability + previous
        linestyle = color_to_linestyle[color]
        if len(probabilities) == 2:
            epislon_height = 0.06
            if probabilities[0] > probabilities[1]:
                y_start = epislon_height if j == 0 else -epislon_height
                x_start = -epislon_height if j == 0 else epislon_height
            else:
                y_start = epislon_height if j == 0 else -epislon_height
                x_start = epislon_height if j == 0 else -epislon_height
            epislon = 0.08
            number_lines = standard_number_lines // 2
            if probability < 0.18:
                x_start *= 2
            if probability < 0.05:
                x_start *= 1.5
                y_start /= 2
                epislon /= 2
                number_lines = standard_number_lines // 8

        else:
            epislon = 0
            y_start = 0
            x_start = 0
            number_lines = standard_number_lines
        x = [x_start] + np.cos(np.linspace(previous + epislon, this - epislon, number_lines)).tolist() + [x_start]
        y = [y_start] + np.sin(np.linspace(previous + epislon, this - epislon, number_lines)).tolist() + [y_start]
        # Potentially remove the lines to the center, if there is single
        if probability in [0, 1]:
            x = x[1:-1]
            y = y[1:-1]
        xy = np.column_stack([x, y])
        previous = this
        markers.append({'marker': xy, 's': np.abs(xy).max() ** 2 * np.array(sizes), 'edgecolors': "k",
                        "facecolor": pie_color, "linewidths": linewidth, "linestyle": linestyle})

    # scatter each of the pie pieces to create pies
    for marker in markers:
        ax.scatter(xs, ys, **marker)


def plot_transition_lines(visualizer, return_period_to_paths, relative_change):
    ax = plt.gca()

    colors = ['k', 'forestgreen', 'limegreen', 'yellowgreen', 'greenyellow', 'palegreen', 'khaki']
    for color, (return_period, paths) in zip(colors[::-1], list(return_period_to_paths.items())):
        x, y = [], []
        for path in paths:
            v = path.vertices
            xv = v[:, 0]
            yv = v[:, 1]
            if (len(x) == 0) or (xv[0] > x[-1]):
                x.extend(xv)
                y.extend(yv)
        if isinstance(return_period, int):
            label = "{}-year return levels".format(return_period)
        else:
            label = 'Mean annual maxima'
        if len(y) > 0:
            value_at_start, value_at_end = y[0], y[-1]
            print(label, value_at_start, value_at_end)
        ax.plot(x, y, label=label, color=color,
                linewidth=3)
    ax.legend(loc='lower right')

    legend_fontsize = 10
    ax.grid()
    ax.set_ylim((2100, 3600))
    ax.set_yticks([int(t) for t in ax.get_yticks() if t % 100 == 0])
    covariates_to_show = [1.5, 2, 2.5, 3, 3.5, 4]
    ax.set_xlim((covariates_to_show[0], covariates_to_show[-1]))
    ax.set_xticks(covariates_to_show)

    add_years_on_top(ax)

    # ax.set_yticklabels(["{} m".format(tic) for tic in ax.get_yticks()])
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show])
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation thresholds, i.e. where the average relative change\n'
                  'with respect to +1 degree is equal to 0\% (m)', fontsize=legend_fontsize)
    visualizer.plot_name = 'Transition lines with relative change = {} and lowest return period {}'.format(
        relative_change, list(return_period_to_paths.keys())[0])
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()


def add_years_on_top(ax):
    years = get_closest_year(AdamontScenario.rcp85_extended, ax.get_xticks())
    ax2 = ax.twiny()
    ax2.set_xlabel('Year')
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(years)


def plot_contour_changes_values(visualizer_list, relative_change, return_period, snowfall):
    default_return_period = OneFoldFit.return_period
    OneFoldFit.return_period = return_period

    ax = plt.gca()
    altitudes = [v.study.altitude for v in visualizer_list]

    covariates = np.linspace(1.5, 4, 50)
    covariates_to_show = [1.5, 2, 2.5, 3, 3.5, 4]

    contour_data = pd.DataFrame()
    for visualizer in visualizer_list:
        altitude = visualizer.study.altitude
        values = get_y(altitude, covariates, None, relative_change, visualizer, return_period)
        df2 = pd.DataFrame({'x': covariates, 'y': [altitude for _ in covariates], 'z': values})
        contour_data = pd.concat([contour_data, df2], axis=0)

    Z = contour_data.pivot_table(index='x', columns='y', values='z').T.values

    X_unique = np.sort(contour_data.x.unique())
    Y_unique = np.sort(contour_data.y.unique())
    X, Y = np.meshgrid(X_unique, Y_unique)

    # Define levels in z-axis where we want lines to appear

    levels = np.array(load_levels(snowfall)[0])

    # Generate a color mapping of the levels we've specified
    cmap, label, norm, _, ticks_values_and_labels, vmax, vmin = load_colorbar_info(relative_change, return_period,
                                                                                   snowfall)

    cpf = ax.contourf(X, Y, Z, len(levels), cmap=cmap, levels=levels)

    # Set all level lines to black
    line_colors = ['black' for _ in cpf.levels]

    # Make plot and customize axes
    cp = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
    level_to_str = {level: '{} \%'.format(int(level)) for level in levels}
    ax.clabel(cp, fontsize=10, colors=line_colors, fmt=level_to_str)
    # ax.clabel(cp, fontsize=10, colors=line_colors, fmt='%.0f')
    legend_fontsize = 10
    ax.set_xticks(covariates_to_show)
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=legend_fontsize)

    ax.set_yticks(altitudes)
    # ax.set_yticklabels(["{} m".format(tic) for tic in ax.get_yticks()])
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show])

    create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=ticks_values_and_labels,
                          fontsize=10, position='top', rescale_ticks=True)

    visualizer = visualizer_list[0]
    visualizer.plot_name = 'Contour plot with relative change = {} with return period {}'.format(relative_change,
                                                                                                 OneFoldFit.return_period)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    OneFoldFit.return_period = default_return_period
    plt.close()

    # Return the paths that corresponds to the level of interest
    return cp.collections[list(levels).index(0)].get_paths()


def plot_piechart_scatter_plot(visualizer_list, all_massif_names, covariates, relative_change, return_period, snowfall):
    ax = plt.gca()
    if len(visualizer_list) == 5:
        # Optimal size for 5 elevations
        sizes = [1500 for _ in range(2)]
    elif len(visualizer_list) == 6:
        sizes = [1000, 100]
    elif len(visualizer_list) == 10:
        sizes = [300 for _ in range(2)]
    else:
        sizes = [1500 for _ in range(2)]

    with_return_level = isinstance(return_period, int)

    color_to_linestyle = {
        'blue': 'dotted',
        'red': 'dashed',
    }
    linewidth_pie = 1

    # color bar
    cmap, label, norm, prefix_label, ticks_values_and_labels, vmax, vmin = load_colorbar_info(relative_change,
                                                                                              return_period, snowfall)

    altitudes = []
    list_nb_valid_massifs = []
    for visualizer in visualizer_list:
        altitude = visualizer.study.altitude
        altitudes.append(altitude)

        # Load facecolors
        values = get_y(altitude, covariates, None, relative_change, visualizer, return_period)
        facecolors = get_colors(values, cmap, vmin, vmax)

        for facecolor, covariate in zip(facecolors, covariates):
            OneFoldFit.COVARIATE_AFTER_TEMPERATURE = covariate
            nb_valid_massifs, *all_l = visualizer.all_trends(all_massif_names.copy(), with_significance=False,
                                                             with_return_level=with_return_level)
            if covariate == covariates[0]:
                list_nb_valid_massifs.append(nb_valid_massifs)
            decreasing_percentage = all_l[0] / 100
            if decreasing_percentage == 1:
                probabilities = [1]
                colors = ['blue']
            elif decreasing_percentage == 0:
                probabilities = [1]
                colors = ['red']
            else:

                probabilities = [decreasing_percentage,
                                 (1 - decreasing_percentage)]
                colors = ['blue', 'red']

            drawPieMarker(ax, covariate, altitude, probabilities, sizes, colors, facecolor,
                          color_to_linestyle, linewidth_pie)

            # add text with the percentage
            if decreasing_percentage > 0.95:
                epsilon = 50
                righ_shift = 0.045
            else:
                righ_shift = 0.055
                epsilon = 40
            shift = -20
            fontsize = 6
            if decreasing_percentage in [0, 1]:
                ax.text((covariate - 0.045),
                        shift + altitude,
                        "100\%",
                        fontsize=fontsize, weight='bold')
            else:
                rounded_decreasing_percentage = round(decreasing_percentage, 2)
                for j, probability in enumerate([1 - rounded_decreasing_percentage, rounded_decreasing_percentage]):
                    ax.text((covariate + righ_shift),
                            shift + altitude - epsilon + 2 * j * epsilon,
                            str(int(100 * probability)) + "\%",
                            fontsize=fontsize, weight='bold')

    legend_fontsize = 10
    labelsize = 10
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation (m)', fontsize=legend_fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.set_xticks([c for c in covariates])
    ax.set_yticks(altitudes)
    # ax.set_yticklabels(["{} m".format(v.study.altitude) for v in visualizer_list])
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates])

    mi, ma = ax.get_ylim()
    border = 100
    ax.set_ylim((mi - border, ma + border + 225))




    # Build legend
    # custom_lines = [Line2D([0], [0], color=color, lw=6) for color in ['blue', 'red']]
    custom_lines = [Patch(facecolor='white', edgecolor='k', linestyle=color_to_linestyle[color],
                          linewidth=linewidth_pie) for color in ['blue', 'red']]
    metric = '{}-year return levels'.format(OneFoldFit.return_period) if with_return_level else 'mean annual maxima'
    labels = ['\% of massifs with {} in {} between +T and +1 degrees'.format(s, metric) for s in
              ['a decrease', 'an increase']]
    ax.legend(custom_lines, labels, prop={'size': 8.5}, loc='upper left')

    create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=ticks_values_and_labels,
                          fontsize=10, position='top', rescale_ticks=True)

    ax2 = ax.twinx()
    ax2.set_ylabel('Number of massifs for each elevation')
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(ax.get_yticks())
    ax2.set_yticklabels([str(nb) for nb in list_nb_valid_massifs])

    label = "return level" if with_return_level else "mean"
    visualizer = visualizer_list[0]
    visualizer.plot_name = 'Piece charts for {} with {}'.format(label, prefix_label)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def load_levels(snowfall, withcolorbar=True):
    if snowfall is None:
        level_max = 20
        graduation = 5
        levels = [0 + graduation * i for i in range(4 + 1)]
    elif snowfall is False:
        level_max = 50
        graduation = 5
        levels = [-level_max + graduation * i for i in range(10 + 1)]
    else:
        if withcolorbar:
            level_max = 20
        else:
            level_max = 40
        graduation = 2
        levels = [-level_max + graduation * i for i in range(level_max + 1)]
    return levels, level_max, graduation


def load_colorbar_info(relative_change, return_period, snowfall, massif_name=None):
    _, max_level, graduation = load_levels(snowfall)
    epsilon = 0.0001
    max_level = max_level + epsilon
    cmap = plt.cm.seismic
    if snowfall:
        ticks_values_and_labels, vmax, vmin = postive_and_negative_values(graduation, max_level)
    elif snowfall is None:
        cmap, ticks_values_and_labels, vmax, vmin = positive_values(cmap, epsilon, graduation, max_level)
    else:
        cmap, ticks_values_and_labels, vmax, vmin = negative_values(cmap, epsilon, graduation, max_level)

    # cmap = get_shifted_map(vmin, vmax, cmap=cmap)
    norm = Normalize(vmin, vmax)
    if massif_name is None:
        prefix_label = 'Average relative change' if relative_change else "Average change"
    else:
        prefix_label = 'Relative change' if relative_change else "Change"
    if isinstance(return_period, int):
        label = prefix_label + ' in {}-year return levels'.format(
            OneFoldFit.return_period)
    else:
        label = prefix_label + ' in mean annual maxima'
    if massif_name is not None:
        label += ' for the {} massif with\nrespect to'.format(massif_name.replace('_', '-'))
    else:
        label += ' with respect to\n'
    label += ' + 1 degree of global warming above pre-industrial levels (\%)'
    return cmap, label, norm, prefix_label, ticks_values_and_labels, vmax, vmin


def postive_and_negative_values(graduation, max_level):
    vmax, vmin = max_level, -max_level
    ticks_values_and_labels = ticks_values_and_labels_for_percentages(graduation, max_level)
    return ticks_values_and_labels, vmax, vmin


def positive_values(cmap, epsilon, graduation, max_level):
    vmax, vmin = max_level, epsilon
    cmap = get_upper_half_colormap(cmap)
    ticks_values, ticks_labels = ticks_values_and_labels_for_percentages(graduation, max_level)
    ticks_values_and_labels = [t * 2 for t in ticks_values[:1 + (len(ticks_labels) // 2)]], ticks_labels[
                                                                                            len(ticks_labels) // 2:]
    return cmap, ticks_values_and_labels, vmax, vmin


def negative_values(cmap, epsilon, graduation, max_level):
    cmap = get_lower_half_colormap(cmap)
    vmax, vmin = -epsilon, -max_level
    ticks_values, ticks_labels = ticks_values_and_labels_for_percentages(graduation, max_level)
    ticks_values_and_labels = [t * 2 for t in ticks_values[:1 + (len(ticks_labels) // 2)]], ticks_labels[:1 + (
            len(ticks_labels) // 2)]
    return cmap, ticks_values_and_labels, vmax, vmin


def plot_relative_change_at_massif_level(visualizer_list, massif_name, with_return_level,
                                         with_significance, relative_change, return_period, snowfall,
                                         temperature_covariate=True):
    legend_fontsize = 10
    labelsize = 10

    default_return_period = OneFoldFit.return_period
    if with_return_level:
        OneFoldFit.return_period = return_period
    cmap = plt.cm.gnuplot
    # colors = cmap(np.linspace(0, 1, len(visualizer_list)))
    colors = ['k', 'darkmagenta', 'darkviolet', 'blueviolet', 'plum', 'pink']
    # colors = ['k', 'forestgreen', 'limegreen', 'yellowgreen', 'greenyellow']
    if temperature_covariate:
        covariates_to_show = [1.5, 2, 2.5, 3, 3.5, 4]
        covariates = np.linspace(covariates_to_show[0], covariates_to_show[-1], num=100)
    else:
        covariates_to_show = [2030 + 10 * i for i in range(8)]
        covariates = [2030 + i for i in range(71)]

    ax = plt.gca()
    all_y = []
    for visualizer in visualizer_list[::-1]:
        altitude = visualizer.study.altitude
        label = '{} m'.format(altitude)
        visualizer: AltitudesStudiesVisualizerForNonStationaryModels

        y = get_y(altitude, covariates, massif_name, relative_change, visualizer, return_period, temperature_covariate)
        if y is not None:
            all_y.extend(y)
            ax.plot(covariates, y, label=label, linewidth=3, color='k')

            index_where_to_plot_text = -10
            covariate_for_text = covariates[index_where_to_plot_text]
            y_for_text = y[index_where_to_plot_text]
            if return_period is not None:
                if (snowfall is True) and (altitude == 1800):
                    y_for_text += 0.75
                if (snowfall is True) and (altitude == 1200):
                    y_for_text += 0.25
                if (snowfall is True) and (altitude == 2100):
                    y_for_text += 0.25
                if (snowfall is False) and (altitude == 1800):
                    y_for_text += 2
                if (snowfall is None) and (altitude == 3000):
                    y_for_text += 1
            if return_period is None:
                if (snowfall is True) and (altitude == 1500):
                    y_for_text += 1

            ax.text(covariate_for_text, y_for_text, label,
                    size=labelsize, color='k', ha="center", va="center", bbox=dict(ec='1', fc='1'))

    if len(all_y) > 0:
        massif_name_str, massif_name_str_2 = get_massif_name_strs(massif_name)
        if temperature_covariate:
            xlabel = 'Global warming above pre-industrial levels ($^o\\textrm{C}$)'
            xtickslabels = ["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show]
        else:
            xlabel = "Years"
            xtickslabels = covariates_to_show
        ax.set_xlabel(xlabel, fontsize=legend_fontsize)
        ax.set_xticks(covariates_to_show)
        ax.set_xticklabels(xtickslabels)
        ax.set_xlim((covariates_to_show[0], covariates_to_show[-1]))

        change_str = 'Relative change' if relative_change else 'Change'
        metric = "{}-year return level".format(OneFoldFit.return_period) if with_return_level else "mean annual maxima"
        ax.tick_params(axis='both', which='major', labelsize=labelsize)


        # Add colorbar on the right
        levels, _, graduation = load_levels(snowfall, withcolorbar=False)
        ax.set_yticks(levels)
        # ax.set_yticklabels([])
        ax.yaxis.grid()
        # cmap, label, norm, prefix_label, ticks_values_and_labels, vmax, vmin = load_colorbar_info(relative_change,
        #                                                                                           return_period, snowfall,
        #                                                                                           massif_name, vmin, vmax)
        cmap, label, *_ = load_colorbar_info(relative_change, return_period, snowfall, massif_name)
        # create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels, rescale_ticks=True, snowfall=snowfall,
        #                       position='right', fontsize=legend_fontsize, vmin=vmin, vmax=vmax)
        ax.set_ylabel(label)
        if temperature_covariate:
            add_years_on_top(ax)

        miny = int(math.floor(min(all_y) / graduation)) * graduation
        maxy = int(math.ceil(max(all_y) / graduation)) * graduation

        ax.set_ylim(miny, maxy)

        # ax.legend(prop={'size': 12}, loc='lower left')

        visualizer = visualizer_list[0]
        visualizer.plot_name = '{}/{} of {} for {} with temp covariate {}'.format(massif_name_str, change_str, metric, massif_name_str, temperature_covariate)
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

        OneFoldFit.return_period = default_return_period

        plt.close()


def plot_relative_change_at_massif_level_sensitivity_to_frequency(
        visualizer: AltitudesStudiesVisualizerForNonStationaryModels, massif_name,
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
        y = get_y(altitude, covariates, massif_name, relative_change, visualizer)
        if y is not None:
            assert len(covariates) == len(y)
            ax.plot(covariates, y, label=label, color=color, linewidth=3)

    legend_fontsize = 10
    labelsize = 10
    massif_name_str, massif_name_str_2 = get_massif_name_strs(massif_name)

    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    change_str = 'Relative change' if relative_change else 'Change'
    unit = '\%' if relative_change else visualizer.study.variable_unit
    metric = "return levels".format(OneFoldFit.return_period)

    ylabel = '{} in {} w.r.t +1 degree of global warming\n{} at {} m ({})'.format(change_str, metric,
                                                                                  massif_name_str_2,
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


def get_massif_name_strs(massif_name):
    massif_name_str = massif_name.replace('_', '-') if massif_name is not None else "All"
    massif_name_str_2 = 'for the {} massif'.format(
        massif_name_str) if massif_name is not None else "averaged on all massifs"
    return massif_name_str, massif_name_str_2


def get_y(altitude, covariates, massif_name, relative_change, visualizer, return_period, temperature_covariate=True):
    order = None if isinstance(return_period, int) else True

    if not temperature_covariate:
        d = year_to_averaged_global_mean_temp(AdamontScenario.rcp85_extended, covariates[0], covariates[-1])
        # Transform time covariate into temperature covariates
        covariates = [d[year] for year in covariates]

    if massif_name is not None:
        if massif_name in visualizer.massif_name_to_one_fold_fit:
            y = get_moments(altitude, covariates, massif_name, relative_change, visualizer, order)
        else:
            y = None
    else:
        y_list = [get_moments(altitude, covariates, massif_name, relative_change, visualizer, order)
                  for massif_name in visualizer.massif_name_to_one_fold_fit.keys()]
        y = np.mean(y_list, axis=0)
    return y


def get_moments(altitude, covariates, massif_name, relative_change, visualizer, order):
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    f = one_fold_fit.relative_changes_of_moment if relative_change else one_fold_fit.changes_of_moment
    return [f([altitude], order, 1, c)[0] for c in covariates]
