import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator

from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_averaged_global_mean_temp, \
    get_closest_year
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import get_colors, \
    create_colorbase_axis, ticks_values_and_labels_for_percentages, get_lower_half_colormap, get_upper_half_colormap, \
    get_inverse_colormap
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projected_extremes.section_results.utils.plot_utils import add_suffix_label


def plot_transition_lines(visualizer, return_period_to_paths, relative_change, legend_fontsize, ticksize):
    ax = plt.gca()

    colors = ['k', 'darkblue', 'mediumblue', 'royalblue', 'cornflowerblue', 'deepskyblue', 'cyan']
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
    ax.legend(loc='lower right', prop={'size': 10})

    ax.grid()
    ax.set_ylim((2100, 3600))
    ax.set_yticks([int(t) for t in ax.get_yticks() if t % 100 == 0])
    covariates_to_show = [1.5, 2, 2.5, 3, 3.5, 4]
    ax.set_xlim((covariates_to_show[0], covariates_to_show[-1]))
    ax.set_xticks(covariates_to_show)

    ax.set_yticklabels(["{}".format(tic) for tic in ax.get_yticks()], fontsize=ticksize)
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show], fontsize=ticksize)
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    ax.set_ylabel('Elevation threshold (m)', fontsize=legend_fontsize)
    visualizer.plot_name = 'Transition lines with relative change = {} and lowest return period {}'.format(
        relative_change, list(return_period_to_paths.keys())[0])
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()


def plot_contour_changes_values(visualizer_list, relative_change, return_period, snowfall, legend_fontsize, ticksize
                                ):
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
    print('here')
    print(levels)
    cp = ax.contour(X, Y, Z, levels=levels, colors=line_colors)
    level_to_str = {level: '{} \%'.format(int(level)) for level in levels}
    ax.clabel(cp, fontsize=10, colors=line_colors, fmt=level_to_str)
    # ax.clabel(cp, fontsize=10, colors=line_colors, fmt='%.0f')
    ax.set_xticks(covariates_to_show)
    ax.set_xlabel('Global warming above pre-industrial levels ($^o\\textrm{C}$)', fontsize=legend_fontsize)
    set_top_label(ax, legend_fontsize, return_period)

    ax.set_ylabel('Elevation (m)', fontsize=legend_fontsize)

    ax.set_yticks(altitudes)
    ax.set_yticklabels(["{}".format(tic) for tic in ax.get_yticks()], fontsize=ticksize)
    ax.set_xticklabels(["+{}".format(int(c) if int(c) == c else c) for c in covariates_to_show], fontsize=ticksize)

    create_colorbase_axis(ax, label, cmap, norm, ticks_values_and_labels=ticks_values_and_labels,
                          fontsize=legend_fontsize, position='top', rescale_ticks=True, ticksize=ticksize)

    visualizer = visualizer_list[0]
    visualizer.plot_name = 'Contour plot with relative change = {} with return period {}'.format(relative_change,
                                                                                                 OneFoldFit.return_period)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    OneFoldFit.return_period = default_return_period
    plt.close()

    # Return the paths that corresponds to the level of interest
    return cp.collections[list(levels).index(0)].get_paths()


def set_top_label(ax, legend_fontsize, return_period, change_str="Relative change"):
    if isinstance(return_period, int):
        top_label = '$\\bf{(b)}$ ' + change_str + ' in ' + str(OneFoldFit.return_period) + '-year return levels'
    else:
        top_label = '$\\bf{(a)}$ ' + change_str + ' in mean annual maxima'
    ax_twin = ax.twiny()
    ax_twin.set_xticks([])
    # ax_twin.set_yticks([])
    ax_twin.set_xlabel(top_label, fontsize=legend_fontsize)

def load_levels(snowfall, withcolorbar=True):
    if snowfall is None:
        level_max = 80
        graduation = 10
        levels = [0 + graduation * i for i in range(8 + 1)]
    elif snowfall is False:
        level_max = 50
        graduation = 5
        levels = [-level_max + graduation * i for i in range(10 + 1)]
    else:
        if withcolorbar:
            level_max = 30
        else:
            level_max = 40
        graduation = 5
        nb_levels = (level_max//graduation) * 2 + 1
        levels = [-level_max + graduation * i for i in range(nb_levels)]
    return levels, level_max, graduation


def load_colorbar_info(relative_change, return_period, snowfall, massif_name=None):
    _, max_level, graduation = load_levels(snowfall)
    epsilon = 0.0001
    max_level = max_level + epsilon
    # cmap = get_inverse_colormap(plt.cm.seismic)
    cmap = plt.cm.BrBG
    if snowfall:
        graduation = 5
        ticks_values_and_labels, vmax, vmin = postive_and_negative_values(graduation, max_level)
    elif snowfall is None:
        cmap, ticks_values_and_labels, vmax, vmin = positive_values(cmap, epsilon, graduation, max_level)
    else:
        cmap, ticks_values_and_labels, vmax, vmin = negative_values(cmap, epsilon, graduation, max_level)

    norm = Normalize(vmin, vmax)
    if massif_name is None:
        prefix_label = 'Average relative change' if relative_change else "Average change"
    else:
        prefix_label = 'Relative change' if relative_change else "Change"
    label = prefix_label
    # if isinstance(return_period, int):
    #     label += ' in {}-year return levels'.format(
    #         OneFoldFit.return_period)
    # else:
    #     label += ' in mean annual maxima'
    label = add_suffix_label(label, massif_name, relative_change)
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
                                         relative_change, return_period, snowfall,
                                         temperature_covariate=True,
                                         categories_list_color=None, legend_fontsize=16,
                                         ticksize=10, nb_days=1):
    inside_size = 9
    colors = ['darkgoldenrod', 'darkgrey', 'mediumseagreen']
    labels_colors = ['Decrease', "Increase followed by a decrease", "Increase"]
    default_return_period = OneFoldFit.return_period
    if with_return_level:
        OneFoldFit.return_period = return_period

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
            if categories_list_color is None:
                color = 'k'
            else:
                assert any([altitude in l for l in categories_list_color])
                assert len(categories_list_color) == 3
                if altitude in categories_list_color[0]:
                    color = colors[0]
                elif altitude in categories_list_color[1]:
                    color = colors[1]
                else:
                    color = colors[2]
            ax.plot(covariates, y, label=label, linewidth=3, color=color)

            # Create a legend
            if categories_list_color is not None:
                legend_elements = [Line2D([0], [0], color=color, lw=4, label=label_color)
                                   for color, label_color in zip(colors[::-1], labels_colors[::-1])]
                ax.legend(handles=legend_elements, loc='lower left', prop={'size': legend_fontsize-2})

            index_where_to_plot_text = -10
            covariate_for_text = covariates[index_where_to_plot_text]
            y_for_text = y[index_where_to_plot_text]
            if return_period is not None:
                if (snowfall is True) and (altitude == 1800):
                    y_for_text += 0.75
                if (snowfall is True) and (altitude == 1200):
                    y_for_text -= 0.75
                    if return_period is None:
                        y_for_text -= 2
                        print('here', return_period, 'adapt plot utils')
                    else:
                        print('here', return_period)
                if (snowfall is True) and (altitude == 1500) and (return_period is None):
                    y_for_text += 0.8
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
                    size=inside_size, color='k', ha="center", va="center", bbox=dict(ec='1', fc='1'))

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
        ax.tick_params(axis='both', which='major', labelsize=ticksize)

        # Add colorbar on the right
        levels, _, graduation = load_levels(snowfall, withcolorbar=False)
        ax.set_yticks(levels)

        cmap, label, *_ = load_colorbar_info(relative_change, return_period, snowfall, massif_name)

        ax.set_ylabel(label, fontsize=legend_fontsize)

        miny = int(math.floor(min(all_y) / graduation)) * graduation
        if snowfall is None:
            maxy = 80
        else:
            maxy = int(math.ceil(max(all_y) / graduation)) * graduation

        if snowfall is True:
            if relative_change:
                miny, maxy = -30, 10
                if nb_days > 1:
                    miny = -35
            else:
                miny, maxy = -15, 15

        ax.set_ylim(miny, maxy)
        ax.yaxis.grid()
        set_top_label(ax, legend_fontsize, return_period, change_str)

        visualizer = visualizer_list[0]
        visualizer.plot_name = '{}/{} of {} for {} with temp covariate {}'.format(massif_name_str, change_str, metric,
                                                                                  massif_name_str,
                                                                                  temperature_covariate)
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
