from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from experiment.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR, EUROCODE_ALTITUDES
from experiment.paper_past_snow_loads.result_trends_and_return_levels.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from experiment.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from root_utils import get_display_name_from_object_type


def plot_uncertainty_massifs(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """ Plot several uncertainty plots
    :return:
    """
    altitude_to_visualizer = {a:v for a,v in altitude_to_visualizer.items() if a in EUROCODE_ALTITUDES}
    visualizer = list(altitude_to_visualizer.values())[-1]
    # Subdivide massif names in group of 3
    m = 1
    uncertainty_massif_names = visualizer.uncertainty_massif_names
    n = (len(uncertainty_massif_names) // m)
    print('total nb of massif', n)
    for i in list(range(n))[:]:
        massif_names = uncertainty_massif_names[m * i: m * (i + 1)]
        print(massif_names)
        plot_subgroup_uncertainty_massifs(altitude_to_visualizer, massif_names)


def plot_subgroup_uncertainty_massifs(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                      massif_names):
    """Create a plot with a maximum of 4 massif names
    We will save the plot at this level
    """
    visualizer = list(altitude_to_visualizer.values())[0]
    nb_massif_names = len(massif_names)
    assert nb_massif_names <= 5
    axes = create_adjusted_axes(nb_massif_names, visualizer.nb_contexts)
    if nb_massif_names == 1:
        axes = [axes]
    for ax, massif_name in zip(axes, massif_names):
        plot_single_uncertainty_massif(altitude_to_visualizer,
                                       massif_name, ax)

    # Save plot
    massif_names_str = '_'.join(massif_names)
    model_names_str = 'NonStationarity=' + '_'.join([str(e) for e in visualizer.non_stationary_contexts])
    visualizer.plot_name = model_names_str + '_' + massif_names_str
    visualizer.show_or_save_to_file(no_title=True)


def plot_single_uncertainty_massif(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                   massif_name, axes):
    visualizer = list(altitude_to_visualizer.values())[0]
    if visualizer.nb_contexts == 1:
        axes = [axes]
    for ax, non_stationary_context in zip(axes, visualizer.non_stationary_contexts):
        plot_single_uncertainty_massif_and_non_stationary_context(ax, massif_name, non_stationary_context,
                                                                  altitude_to_visualizer)


def get_label_name(non_stationary_context, ci_method_name):
    model_symbol = 'N' if non_stationary_context else '0'
    parameter = ', 2017' if non_stationary_context else ''
    model_name = ' $ \widehat{z_p}(\\boldsymbol{\\theta_{\mathcal{M}_'
    model_name += model_symbol
    model_name += '}}'
    model_name += parameter
    model_name += ')_{ \\textrm{' + ci_method_name.upper().split(' ')[1] + '}} $ '
    return model_name


def plot_single_uncertainty_massif_and_non_stationary_context(ax, massif_name, non_stationary_context,
                                                              altitude_to_visualizer: Dict[
                                                                  int, StudyVisualizerForNonStationaryTrends]):
    """ Generic function that might be used by many other more global functions"""
    altitudes = list(altitude_to_visualizer.keys())
    visualizer = list(altitude_to_visualizer.values())[0]
    colors = ['tab:green', 'tab:brown'][::-1]
    alpha = 0.2
    # Display the EUROCODE return level
    eurocode_region = massif_name_to_eurocode_region[massif_name]()

    # Display the return level from model class
    for j, (color, uncertainty_method) in enumerate(zip(colors, visualizer.uncertainty_methods)):
        if j == 0:
            # Plot eurocode norm
            eurocode_region.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax,
                                                                                                   altitudes=altitudes)

        # Plot uncertainties
        valid_altitudes = plot_valid_return_level_uncertainties(alpha, altitude_to_visualizer, altitudes, ax, color,
                                                                massif_name, non_stationary_context, uncertainty_method)

        # Plot bars of TDRL only in the non stationary case
        if j == 0 and non_stationary_context:
            plot_tdrl_bars(altitude_to_visualizer, ax, massif_name, valid_altitudes)

    ax.legend(loc=2)
    ax.set_ylim([-1, 16])
    massif_name_str = massif_name.replace('_', ' ')
    eurocode_region_str = get_display_name_from_object_type(type(eurocode_region))
    is_non_stationary_model = non_stationary_context if isinstance(non_stationary_context,
                                                                   bool) else 'Non' in non_stationary_context
    if is_non_stationary_model:
        non_stationary_context = 'non-stationary'
    else:
        non_stationary_context = 'stationary'
    title = '{} ({} Eurocodes area) with a {} model'.format(massif_name_str, eurocode_region_str,
                                                            non_stationary_context)
    ax.set_title(title)
    ax.set_xticks(altitudes)
    ax.set_ylabel(EUROCODE_RETURN_LEVEL_STR)
    ax.set_xlabel('Altitude (m)')
    ax.grid()


def plot_tdrl_bars(altitude_to_visualizer, ax, massif_name, valid_altitudes):
    visualizers = [v for a, v in altitude_to_visualizer.items() if a in valid_altitudes and massif_name in v.uncertainty_massif_names]
    if len(visualizers) > 0:
        tdrl_values = [v.massif_name_to_tdrl_value[massif_name] for v in visualizers]
        # Plot bars
        colors = [v.massif_name_to_tdrl_color[massif_name] for v in visualizers]
        ax.bar(valid_altitudes, tdrl_values, width=150, color=colors, label=visualizers[0].label_tdrl_bar,
               edgecolor='black', hatch='//')
        # Plot markers
        markers_kwargs = [v.massif_name_to_marker_style[massif_name] for v in visualizers]
        for altitude, marker_kwargs, value in zip(valid_altitudes, markers_kwargs, tdrl_values):
            # ax.plot([altitude], [value / 2], **marker_kwargs)
            # Better to plot all the markers on the same line
            ax.plot([altitude], 0, **marker_kwargs)


def plot_valid_return_level_uncertainties(alpha, altitude_to_visualizer, altitudes, ax, color, massif_name,
                                          non_stationary_context, uncertainty_method):
    # Compute ordered_return_level_uncertaines for a given massif_name, uncertainty methods, and non stationary context
    ordered_return_level_uncertainties = []
    for visualizer in altitude_to_visualizer.values():
        u = visualizer.triplet_to_eurocode_uncertainty[(uncertainty_method, non_stationary_context, massif_name)]
        ordered_return_level_uncertainties.append(u)
    # Display
    mean = [r.mean_estimate for r in ordered_return_level_uncertainties]
    # Filter and keep only non nan values
    not_nan_index = [not np.isnan(m) for m in mean]
    mean = list(np.array(mean)[not_nan_index])
    valid_altitudes = list(np.array(altitudes)[not_nan_index])
    ordered_return_level_uncertainties = list(np.array(ordered_return_level_uncertainties)[not_nan_index])
    ci_method_name = str(uncertainty_method).split('.')[1].replace('_', ' ')
    label_name = get_label_name(non_stationary_context, ci_method_name)
    ax.plot(valid_altitudes, mean, linestyle='--', marker='o', color=color,
            label=label_name)
    lower_bound = [r.confidence_interval[0] for r in ordered_return_level_uncertainties]
    upper_bound = [r.confidence_interval[1] for r in ordered_return_level_uncertainties]
    confidence_interval_str = ' {}'.format(AbstractExtractEurocodeReturnLevel.percentage_confidence_interval)
    confidence_interval_str += '\% confidence interval'
    ax.fill_between(valid_altitudes, lower_bound, upper_bound, color=color, alpha=alpha, label=label_name + confidence_interval_str)
    return valid_altitudes
