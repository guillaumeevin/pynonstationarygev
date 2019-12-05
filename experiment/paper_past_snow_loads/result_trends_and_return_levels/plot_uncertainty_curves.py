from typing import Dict
import matplotlib.pyplot as plt

import numpy as np

from experiment.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR, EUROCODE_ALTITUDES
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from experiment.paper_past_snow_loads.paper_utils import dpi_paper1_figure
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from experiment.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import ci_method_to_color
from root_utils import get_display_name_from_object_type


def plot_uncertainty_massifs(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """ Plot several uncertainty plots
    :return:
    """
    altitude_to_visualizer = {a: v for a, v in altitude_to_visualizer.items() if a in EUROCODE_ALTITUDES}
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
    # axes = create_adjusted_axes(nb_massif_names, visualizer.nb_contexts)
    # if nb_massif_names == 1:
    #     axes = [axes]
    for massif_name in massif_names:
        plot_single_uncertainty_massif(altitude_to_visualizer, massif_name)


def plot_single_uncertainty_massif(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                   massif_name):
    visualizer = list(altitude_to_visualizer.values())[0]

    for non_stationary_context in visualizer.non_stationary_contexts:
        ax = create_adjusted_axes(1, 1)
        plot_single_uncertainty_massif_and_non_stationary_context(ax, massif_name, non_stationary_context,
                                                                  altitude_to_visualizer)
        # Save plot
        massif_names_str = massif_name
        model_names_str = 'NonStationarity={}'.format(non_stationary_context)
        visualizer.plot_name = model_names_str + '_' + massif_names_str
        visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
        plt.close()


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
    alpha = 0.2
    legend_size = 25
    fontsize_label = 35
    # Display the EUROCODE return level
    eurocode_region = massif_name_to_eurocode_region[massif_name]()

    # Display the return level from model class
    for j, uncertainty_method in enumerate(visualizer.uncertainty_methods):
        if j == 0:
            # Plot eurocode norm
            eurocode_region.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax,
                                                                                                   altitudes=altitudes)

        # Plot uncertainties
        color = ci_method_to_color[uncertainty_method]
        valid_altitudes = plot_valid_return_level_uncertainties(alpha, altitude_to_visualizer, altitudes, ax, color,
                                                                massif_name, non_stationary_context, uncertainty_method)

        # Plot bars of TDRL only in the non stationary case
        if j == 0 and non_stationary_context:
            plot_tdrl_bars(altitude_to_visualizer, ax, massif_name, valid_altitudes, legend_size, legend_size)

    ax.legend(loc=2, prop={'size': legend_size})
    # ax.set_ylim([-1, 16])
    ax.set_xlim([200, 1900])
    if massif_name == 'Maurienne':
        ax.set_ylim([-1.5, 13])
    # add_title(ax, eurocode_region, massif_name, non_stationary_context)
    ax.set_xticks(altitudes)
    ax.tick_params(labelsize=fontsize_label)
    ylabel = EUROCODE_RETURN_LEVEL_STR.replace('GSL', SCM_STUDY_CLASS_TO_ABBREVIATION[type(visualizer.study)])
    ax.set_ylabel(ylabel, fontsize=fontsize_label)
    ax.set_xlabel('Altitude (m)', fontsize=fontsize_label)
    ax.grid()


def add_title(ax, eurocode_region, massif_name, non_stationary_context):
    massif_name_str = massif_name.replace('_', ' ')
    eurocode_region_str = get_display_name_from_object_type(type(eurocode_region))
    is_non_stationary_model = non_stationary_context if isinstance(non_stationary_context,
                                                                   bool) else 'Non' in non_stationary_context
    if is_non_stationary_model:
        non_stationary_context = 'selected non-stationary models'
    else:
        non_stationary_context = 'the stationary model'
    title = '{} massif with {}'.format(massif_name_str, non_stationary_context)
    ax.set_title(title)


def plot_tdrl_bars(altitude_to_visualizer, ax, massif_name, valid_altitudes, legend_size, fontsize):
    visualizers = [v for a, v in altitude_to_visualizer.items() if
                   a in valid_altitudes and massif_name in v.uncertainty_massif_names]
    if len(visualizers) > 0:
        tdrl_values = [v.massif_name_to_tdrl_value[massif_name] for v in visualizers]
        # Plot bars
        colors = [v.massif_name_to_tdrl_color[massif_name] for v in visualizers]
        ax.bar(valid_altitudes, tdrl_values, width=150, color=colors, label=visualizers[0].label_tdrl_bar,
               edgecolor='black', hatch='//')
        # Plot markers
        markers_kwargs = [v.massif_name_to_marker_style[massif_name] for v in visualizers]
        for k in markers_kwargs:
            k['markersize'] = 7
        for altitude, marker_kwargs, value in zip(valid_altitudes, markers_kwargs, tdrl_values):
            # ax.plot([altitude], [value / 2], **marker_kwargs)
            # Better to plot all the markers on the same line
            ax.plot([altitude], 0, **marker_kwargs)
    # Add a legend plot
    legend_elements = AbstractStudy.get_legend_for_model_symbol(markersize=9)
    ax2 = ax.twinx()
    # ax2.legend(handles=legend_elements, bbox_to_anchor=(0.93, 0.7), loc='upper right')
    # ax2.annotate("Filled symbol = significant trend ", xy=(0.85, 0.5), xycoords='axes fraction', fontsize=7)
    ax2.legend(handles=legend_elements, loc='upper right', prop={'size': legend_size})
    ax2.annotate("Filled symbol =\n significant trend ", xy=(0.6, 0.85), xycoords='axes fraction', fontsize=fontsize)
    ax2.set_yticks([])


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
    ax.fill_between(valid_altitudes, lower_bound, upper_bound, color=color, alpha=alpha,
                    label=label_name + confidence_interval_str)
    return valid_altitudes
