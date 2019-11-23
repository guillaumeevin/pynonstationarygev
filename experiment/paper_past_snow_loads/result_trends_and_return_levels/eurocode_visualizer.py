from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from experiment.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from root_utils import get_display_name_from_object_type


def get_label_name(model_name, ci_method_name: str):
    is_non_stationary = model_name == 'NonStationary'
    model_symbol = 'N' if is_non_stationary else '0'
    parameter = ', 2017' if is_non_stationary else ''
    model_name = ' $ \widehat{z_p}(\\boldsymbol{\\theta_{\mathcal{M}_'
    model_name += model_symbol
    model_name += '}}'
    model_name += parameter
    model_name += ')_{ \\textrm{' + ci_method_name.upper().split(' ')[1] + '}} $ '
    return model_name


def get_model_name(model_class):
    return get_display_name_from_object_type(model_class).split('Stationary')[0] + 'Stationary'

def massif_name_to_ordered_return_level_uncertainties(altitude_to_visualizer, massif_names,
                                                      uncertainty_methods, temporal_covariate,
                                                      non_stationary_model):
    massif_name_to_ordered_eurocode_level_uncertainty = {
        massif_name: {ci_method: [] for ci_method in uncertainty_methods} for massif_name in massif_names}
    for altitude, visualizer in altitude_to_visualizer.items():
        print('Processing altitude = {} '.format(altitude))
        for ci_method in uncertainty_methods:
            d = visualizer.massif_name_to_altitude_and_eurocode_level_uncertainty_for_minimized_aic_model_class(
                massif_names, ci_method,
                temporal_covariate, non_stationary_model)
            # Append the altitude one by one
            for massif_name, return_level_uncertainty in d.items():
                print(massif_name, return_level_uncertainty[0], return_level_uncertainty[1].confidence_interval,
                      return_level_uncertainty[1].mean_estimate)
                massif_name_to_ordered_eurocode_level_uncertainty[massif_name][ci_method].append(
                    return_level_uncertainty)
    return massif_name_to_ordered_eurocode_level_uncertainty


def plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict(altitude_to_visualizer,
                                                                         massif_names,
                                                                         non_stationary_models_for_uncertainty,
                                                                         uncertainty_methods):
    """
    Rows correspond to massif names
    Columns correspond to stationary/non stationary model name for a given date
    Uncertainty result_trends_and_return_levels correpsond to the different plot on the graph
    :return:
    """
    # Compute the dictionary of interest
    # Plot uncertainties
    model_name_to_massif_name_to_ordered_return_level = {}
    for non_stationary_model in non_stationary_models_for_uncertainty:
        d = massif_name_to_ordered_return_level_uncertainties(altitude_to_visualizer, massif_names,
                                                              uncertainty_methods,
                                                              temporal_covariate=2017,
                                                              non_stationary_model=non_stationary_model)
        model_name_to_massif_name_to_ordered_return_level[non_stationary_model] = d

    # Transform the dictionary into the desired format
    d = {}
    for massif_name in massif_names:
        d2 = {model_name: model_name_to_massif_name_to_ordered_return_level[model_name][massif_name] for model_name
              in
              model_name_to_massif_name_to_ordered_return_level.keys()}
        d[massif_name] = d2

    nb_massif_names = len(massif_names)
    nb_model_names = len(non_stationary_models_for_uncertainty)
    axes = create_adjusted_axes(nb_massif_names, nb_model_names)
    if nb_massif_names == 1:
        axes = [axes]
    for ax, (massif_name, model_name_to_uncertainty_level) in zip(axes, d.items()):
        plot_model_name_to_uncertainty_method_to_ordered_dict(model_name_to_uncertainty_level,
                                                              massif_name, ax)

    # Save plot
    visualizer = list(altitude_to_visualizer.values())[0]
    massif_names_str = '_'.join(massif_names)
    model_names_str = 'NonStationarity=' + '_'.join([str(e) for e in non_stationary_models_for_uncertainty])
    visualizer.plot_name = model_names_str + '_' + massif_names_str
    visualizer.show_or_save_to_file(no_title=True)

    # plt.suptitle('50-year return levels of extreme snow loads in France for several confiance interval methods.')

def plot_model_name_to_uncertainty_method_to_ordered_dict(d, massif_name, axes):
    if len(d) == 1:
        axes = [axes]
    for ax, (model_name, uncertainty_method_to_ordered_dict) in zip(axes, d.items()):
        plot_label_to_ordered_return_level_uncertainties(ax, massif_name, model_name,
                                                         uncertainty_method_to_ordered_dict)


def plot_label_to_ordered_return_level_uncertainties(ax, massif_name, model_name,
                                                     label_to_ordered_return_level_uncertainties:
                                                     Dict[str, List[
                                                         EurocodeConfidenceIntervalFromExtremes]]):
    """ Generic function that might be used by many other more global functions"""
    colors = ['tab:green', 'tab:olive']
    alpha = 0.2
    # Display the EUROCODE return level
    eurocode_region = massif_name_to_eurocode_region[massif_name]()

    # Display the return level from model class
    for j, (color, (label, l)) in enumerate(zip(colors, label_to_ordered_return_level_uncertainties.items())):
        l = list(zip(*l))
        altitudes = l[0]
        ordered_return_level_uncertaines = l[1]  # type: List[EurocodeConfidenceIntervalFromExtremes]
        # Plot eurocode standards only for the first loop
        if j == 0:
            eurocode_region.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax, altitudes=altitudes)
        mean = [r.mean_estimate for r in ordered_return_level_uncertaines]
        # Filter and keep only non nan values
        not_nan_index = [not np.isnan(m) for m in mean]
        mean = list(np.array(mean)[not_nan_index])
        altitudes = list(np.array(altitudes)[not_nan_index])
        ordered_return_level_uncertaines = list(np.array(ordered_return_level_uncertaines)[not_nan_index])

        ci_method_name = str(label).split('.')[1].replace('_', ' ')
        ax.plot(altitudes, mean, linestyle='--', marker='o', color=color, label=get_label_name(model_name, ci_method_name))
        lower_bound = [r.confidence_interval[0] for r in ordered_return_level_uncertaines]
        upper_bound = [r.confidence_interval[1] for r in ordered_return_level_uncertaines]
        ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha)
    ax.legend(loc=2)
    ax.set_ylim([0.0, 16])
    massif_name_str = massif_name.replace('_', ' ')
    eurocode_region_str = get_display_name_from_object_type(type(eurocode_region))
    is_non_stationary_model = model_name if isinstance(model_name, bool) else 'Non' in model_name
    if is_non_stationary_model:
        model_name = 'non-stationary'
    else:
        model_name = 'stationary'
    title = '{} ({} Eurocodes area) with a {} model'.format(massif_name_str, eurocode_region_str, model_name)
    ax.set_title(title)
    ax.set_ylabel('50-year return level of SL (kN $m^-2$)')
    ax.set_xlabel('Altitude (m)')
    ax.grid()

