from typing import Dict, List

import matplotlib.pyplot as plt

from experiment.eurocode_data.eurocode_return_level_uncertainties import EurocodeLevelUncertaintyFromExtremes
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES
from experiment.eurocode_data.utils import EUROCODE_QUANTILE, EUROCODE_ALTITUDES
from experiment.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes


def plot_model_name_to_dep_to_ordered_return_level_uncertainties(
        dep_to_model_name_to_ordered_return_level_uncertainties, show=True):
    # Create a 9 x 9 plot
    axes = create_adjusted_axes(3, 3)
    axes = list(axes.flatten())
    ax6 = axes[5]
    ax9 = axes[8]

    axes.remove(ax6)
    axes.remove(ax9)
    ax_to_departement = dict(zip(axes, DEPARTEMENT_TYPES[::-1]))
    for ax, departement in ax_to_departement.items():
        plot_dep_to_model_name_dep_to_ordered_return_level_uncertainties(ax, departement,
                                                                         dep_to_model_name_to_ordered_return_level_uncertainties[
                                                                             departement]
                                                                         )
    ax6.remove()
    ax9.remove()

    plt.suptitle('50-year return levels for all French Alps departements. \n'
                 'Comparison between the maximum EUROCODE in the departement\n'
                 'and the maximum return level found for the massif belonging to the departement')
    if show:
        plt.show()


def plot_dep_to_model_name_dep_to_ordered_return_level_uncertainties(ax, dep_class,
                                                                     model_name_to_ordered_return_level_uncertainties:
                                                                     Dict[str, List[
                                                                         EurocodeLevelUncertaintyFromExtremes]]):
    colors = ['red', 'blue', 'green']
    altitudes = EUROCODE_ALTITUDES
    alpha = 0.2
    # Display the EUROCODE return level
    dep_object = dep_class()
    dep_object.eurocode_region.plot_max_loading(ax, altitudes=altitudes)
    # Display the return level from model class
    for color, (model_name, ordered_return_level_uncertaines) in zip(colors,
                                                                     model_name_to_ordered_return_level_uncertainties.items()):
        mean = [r.posterior_mean for r in ordered_return_level_uncertaines]
        ax.plot(altitudes, mean, '-', color=color)
        lower_bound = [r.poster_uncertainty_interval[0] for r in ordered_return_level_uncertaines]
        upper_bound = [r.poster_uncertainty_interval[1] for r in ordered_return_level_uncertaines]
        ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha)
    ax.set_title(str(dep_object))
    ax.set_ylabel('Maximum {} quantile (in N $m^-2$)'.format(EUROCODE_QUANTILE))
    ax.set_xlabel('Altitude')
