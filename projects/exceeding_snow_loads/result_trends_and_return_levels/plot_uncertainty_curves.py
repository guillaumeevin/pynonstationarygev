from typing import Dict
import matplotlib.pyplot as plt

import numpy as np

from experiment.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR, EUROCODE_ALTITUDES, \
    YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy, filled_marker_legend_list2
from projects.exceeding_snow_loads.paper_utils import dpi_paper1_figure, ModelSubsetForUncertainty
from projects.exceeding_snow_loads.study_visualizer_for_non_stationary_trends import \
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
    nb_massif_names = len(massif_names)
    assert nb_massif_names <= 5
    for massif_name in massif_names:
        plot_single_uncertainty_massif(altitude_to_visualizer, massif_name)


def plot_single_uncertainty_massif(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                   massif_name):
    visualizer = list(altitude_to_visualizer.values())[0]

    model_subsets_for_uncertainty = [ModelSubsetForUncertainty.stationary_gumbel,
            ModelSubsetForUncertainty.non_stationary_gumbel_and_gev]
    print('Subsets for uncertainty curves:{}'.format(model_subsets_for_uncertainty))
    for model_subset_for_uncertainty in model_subsets_for_uncertainty:
        ax = create_adjusted_axes(1, 1)
        plot_single_uncertainty_massif_and_non_stationary_context(ax, massif_name, model_subset_for_uncertainty,
                                                                  altitude_to_visualizer)
        # Save plot
        massif_names_str = massif_name
        model_names_str = get_display_name_from_object_type(model_subset_for_uncertainty)
        visualizer.plot_name = model_names_str + '_' + massif_names_str
        visualizer.show_or_save_to_file(no_title=True, dpi=dpi_paper1_figure)
        plt.close()


def get_label_name(model_subset_for_uncertainty, ci_method_name, add_method_suffix):
    model_symbol = 'N' if model_subset_for_uncertainty is not ModelSubsetForUncertainty.stationary_gumbel else '0'
    parameter = ', {}'.format(YEAR_OF_INTEREST_FOR_RETURN_LEVEL) if model_subset_for_uncertainty not in [ModelSubsetForUncertainty.stationary_gumbel,
                                                                 ModelSubsetForUncertainty.stationary_gumbel_and_gev] \
        else ''
    model_name = ' $ z_p(\\boldsymbol{\\widehat{\\theta}_{\\mathcal{M}'
    # model_name += '_' + model_symbol
    model_name += '}}'
    model_name += parameter
    model_name += ')'
    if add_method_suffix:
        model_name += '_{ \\textrm{' + ci_method_name.upper().split(' ')[1] + '}} '
    model_name += '$'
    return model_name


def plot_single_uncertainty_massif_and_non_stationary_context(ax, massif_name, model_subset_for_uncertainty,
                                                              altitude_to_visualizer: Dict[
                                                                  int, StudyVisualizerForNonStationaryTrends]):
    """ Generic function that might be used by many other more global functions"""
    altitudes = list(altitude_to_visualizer.keys())
    visualizer = list(altitude_to_visualizer.values())[0]
    alpha = 0.2
    legend_size = 30
    fontsize_label = 35
    # Display the EUROCODE return level
    eurocode_region = massif_name_to_eurocode_region[massif_name]()

    # Display the return level from model class
    nb_uncertainty_methods = len(visualizer.uncertainty_methods)
    for j, uncertainty_method in enumerate(visualizer.uncertainty_methods):
        if j == 0:
            # Plot eurocode norm
            altitudes_for_plot = list(range(min(altitudes), max(altitudes)+1, 100))
            eurocode_region.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax,
                                                                                                   altitudes=altitudes_for_plot)

        # Plot uncertainties
        color = ci_method_to_color[uncertainty_method]
        valid_altitudes = plot_valid_return_level_uncertainties(alpha, altitude_to_visualizer, altitudes, ax, color,
                                                                massif_name, model_subset_for_uncertainty, uncertainty_method,
                                                                nb_uncertainty_methods)
        # Plot some data for the non valid altitudes

        # Plot bars of TDRL only in the general non stationary case
        if model_subset_for_uncertainty is ModelSubsetForUncertainty.non_stationary_gumbel_and_gev:
            plot_tdrl_bars(altitude_to_visualizer, ax, massif_name, valid_altitudes, legend_size, legend_size)

    ax.legend(loc=2, prop={'size': legend_size})
    # ax.set_ylim([-1, 16])
    ax.set_xlim([200, 1900])
    if massif_name in ['Maurienne', 'Chartreuse', 'Beaufortain']:
        ax.set_ylim([-1.5, 13.5])
        ax.set_yticks([2 * i for i in range(7)])
    if massif_name in ['Vercors']:
        ax.set_ylim([-1, 10])
        ax.set_yticks([2 * i for i in range(6)])



    # add_title(ax, eurocode_region, massif_name, non_stationary_context)
    ax.set_xticks(altitudes)
    ax.tick_params(labelsize=fontsize_label)
    ylabel = EUROCODE_RETURN_LEVEL_STR
    # ylabel = EUROCODE_RETURN_LEVEL_STR.replace('GSL', SCM_STUDY_CLASS_TO_ABBREVIATION[type(visualizer.study)])
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
    visualizers = [v for a, v in altitude_to_visualizer.items()
                   if a in valid_altitudes and massif_name in v.massif_names_fitted]
    if len(visualizers) > 0:
        tdrl_values = [v.massif_name_to_tdrl_value[massif_name] for v in visualizers]
        # Plot bars
        # colors = [v.massif_name_to_color[massif_name] for v in visualizers]
        # From snow on, we set a black color for the bars
        colors = ['white' for v in visualizers]
        non_null_tdrl_index = [i for i, t in enumerate(tdrl_values) if t != 0.0]
        if len(non_null_tdrl_index) > 0:
            ax.bar(np.array(valid_altitudes)[non_null_tdrl_index], np.array(tdrl_values)[non_null_tdrl_index],
                   width=150, color=colors, label=visualizers[0].label_tdrl_bar, edgecolor='black', hatch='//')
        # Plot markers
        markers_kwargs = [v.massif_name_to_marker_style[massif_name] for v in visualizers]
        markersize = 20
        for k in markers_kwargs:
            k['markersize'] = markersize
        for altitude, marker_kwargs, value in zip(valid_altitudes, markers_kwargs, tdrl_values):
            # ax.plot([altitude], [value / 2], **marker_kwargs)
            # Better to plot all the markers on the same line
            ax.plot([altitude], 0, **marker_kwargs)
        # Add a legend plot
        visualizer = visualizers[0]
        markers = [v.massif_name_to_marker_style[massif_name]['marker'] for v in visualizers]
        marker_to_label = {m: visualizer.all_marker_style_to_label_name[m] for m in markers}
        legend_elements = AbstractStudy.get_legend_for_model_symbol(marker_to_label, markersize=markersize)
        ax2 = ax.twinx()
        # ax2.legend(handles=legend_elements, bbox_to_anchor=(0.93, 0.7), loc='upper right')
        # ax2.annotate("Filled symbol = significant trend ", xy=(0.85, 0.5), xycoords='axes fraction', fontsize=7)
        upper_legend_y = 0.55
        ax2.annotate('\n'.join(filled_marker_legend_list2), xy=(0.23, upper_legend_y - 0.2), xycoords='axes fraction', fontsize=fontsize)
        ax2.annotate('Markers show selected model $\mathcal{M}_N$', xy=(0.02, upper_legend_y), xycoords='axes fraction', fontsize=fontsize)
        print(legend_size)
        ax2.legend(handles=legend_elements, prop={'size': legend_size},
                   loc='upper left', bbox_to_anchor=(0.00, upper_legend_y))
        # for handle in lgnd.legendHandles:
        #     handle.set_sizes([6.0])
        # ax2.annotate("Filled symbol =\nsignificant trend  \nw.r.t $\mathcal{M}_0$", xy=(0.6, 0.85), xycoords='axes fraction', fontsize=fontsize)

        ax2.set_yticks([])


def plot_valid_return_level_uncertainties(alpha, altitude_to_visualizer, altitudes, ax, color, massif_name,
                                          model_subset_for_uncertainty, uncertainty_method, nb_uncertainty_methods):
    # Compute ordered_return_level_uncertaines for a given massif_name, uncertainty methods, and non stationary context
    ordered_return_level_uncertainties = []
    for visualizer in altitude_to_visualizer.values():
        u = visualizer.triplet_to_eurocode_uncertainty[(uncertainty_method, model_subset_for_uncertainty, massif_name)]
        ordered_return_level_uncertainties.append(u)
    # Display
    mean = [r.mean_estimate for r in ordered_return_level_uncertainties]
    # Filter and keep only non nan values
    not_nan_index = [not np.isnan(m) for m in mean]
    mean = list(np.array(mean)[not_nan_index])
    valid_altitudes = list(np.array(altitudes)[not_nan_index])
    ordered_return_level_uncertainties = list(np.array(ordered_return_level_uncertainties)[not_nan_index])
    ci_method_name = str(uncertainty_method).split('.')[1].replace('_', ' ')
    add_method_suffix = nb_uncertainty_methods > 1 or 'mle' not in ci_method_name
    label_name = get_label_name(model_subset_for_uncertainty, ci_method_name, add_method_suffix=add_method_suffix)
    ax.plot(valid_altitudes, mean, linestyle='--', marker='o', color=color,
            label=label_name)
    lower_bound = [r.confidence_interval[0] for r in ordered_return_level_uncertainties]
    upper_bound = [r.confidence_interval[1] for r in ordered_return_level_uncertainties]
    confidence_interval_str = ' {}'.format(AbstractExtractEurocodeReturnLevel.percentage_confidence_interval)
    confidence_interval_str += '\% confidence interval'
    ax.fill_between(valid_altitudes, lower_bound, upper_bound, color=color, alpha=alpha,
                    label=label_name + confidence_interval_str)
    # Plot error bars
    yerr = np.array([[d[1] - d[0], d[2] - d[1]] for d in zip(lower_bound, mean, upper_bound)]).transpose()
    ax.bar(valid_altitudes, mean,  ecolor='black', capsize=5, yerr=yerr)
    return valid_altitudes
