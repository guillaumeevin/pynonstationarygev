from typing import Dict, List

import matplotlib.pyplot as plt

from experiment.eurocode_data.eurocode_return_level_uncertainties import EurocodeLevelUncertaintyFromExtremes
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES, massif_name_to_eurocode_region
from experiment.eurocode_data.utils import EUROCODE_QUANTILE, EUROCODE_ALTITUDES
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from root_utils import get_display_name_from_object_type


def get_label_name(model_name, ci_method_name: str):
    is_non_stationary = model_name == 'NonStationary'
    model_symbol = '{\mu_1, \sigma_1}' if is_non_stationary else '0'
    parameter = ', 2017' if is_non_stationary else ''
    model_name = ' $ \widehat{z_p}(\\boldsymbol{\\theta_{\mathcal{M}_'
    model_name += model_symbol
    model_name += '}}'
    model_name += parameter
    model_name += ')_{ \\textrm{' + ci_method_name.upper() + '}} $ '
    return model_name


def get_model_name(model_class):
    return get_display_name_from_object_type(model_class).split('Stationary')[0] + 'Stationary'

def plot_massif_name_to_model_name_to_uncertainty_method_to_ordered_dict(d, nb_massif_names, nb_model_names, show=True):
    """
    Rows correspond to massif names
    Columns correspond to stationary/non stationary model name for a given date
    Uncertainty method correpsond to the different plot on the graph
    :return:
    """
    axes = create_adjusted_axes(nb_massif_names, nb_model_names)
    if nb_massif_names == 1:
        axes = [axes]
    for ax, (massif_name, model_name_to_uncertainty_level) in zip(axes, d.items()):
        plot_model_name_to_uncertainty_method_to_ordered_dict(model_name_to_uncertainty_level,
                                                              massif_name, ax)

    plt.suptitle('50-year return levels of extreme snow loads in France for several confiance interval methods.')

    if show:
        plt.show()


def plot_model_name_to_uncertainty_method_to_ordered_dict(d, massif_name, axes):
    if len(d) == 1:
        axes = [axes]
    for ax, (model_name, uncertainty_method_to_ordered_dict) in zip(axes, d.items()):
        plot_label_to_ordered_return_level_uncertainties(ax, massif_name, model_name,
                                                         uncertainty_method_to_ordered_dict)


def plot_label_to_ordered_return_level_uncertainties(ax, massif_name, model_name,
                                                     label_to_ordered_return_level_uncertainties:
                                                     Dict[str, List[
                                                         EurocodeLevelUncertaintyFromExtremes]]):
    """ Generic function that might be used by many other more global functions"""
    colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:olive']
    alpha = 0.2
    # Display the EUROCODE return level
    eurocode_region = massif_name_to_eurocode_region[massif_name]()

    # Display the return level from model class
    for j, (color, (label, l)) in enumerate(zip(colors,label_to_ordered_return_level_uncertainties.items())):
        altitudes, ordered_return_level_uncertaines = zip(*l)
        # Plot eurocode standards only for the first loop
        if j == 0:
            eurocode_region.plot_max_loading(ax, altitudes=altitudes)
        mean = [r.posterior_mean for r in ordered_return_level_uncertaines]

        ci_method_name = str(label).split('.')[1].replace('_', ' ')
        ax.plot(altitudes, mean, '-', color=color, label=get_label_name(model_name, ci_method_name))
        lower_bound = [r.poster_uncertainty_interval[0] for r in ordered_return_level_uncertaines]
        upper_bound = [r.poster_uncertainty_interval[1] for r in ordered_return_level_uncertaines]
        ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha)
    ax.legend(loc=2)
    ax.set_ylim([0.0, 4.0])
    ax.set_title(massif_name + ' ' + model_name)
    ax.set_ylabel('50-year return level (N $m^-2$)')
    ax.set_xlabel('Altitude (m)')
