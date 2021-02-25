import matplotlib.pyplot as plt
import numpy as np

from extreme_data.eurocode_data.eurocode_region import E, C
from extreme_data.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from extreme_data.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from root_utils import get_display_name_from_object_type


def main_eurocode_norms(ax=None, poster_plot=False):
    if ax is None:
        ax = plt.gca()
        altitudes = np.linspace(200, 2000, 1800 + 1)
        for region_class in [E, C]:
            region_object = region_class()
            label = get_display_name_from_object_type(region_class) + ' ' + 'region'
            region_object.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax, altitudes,
                                                                                                 label=label,
                                                                                                 linestyle='-')
            if region_class == C:
                ax.xaxis.set_ticks([250 * i for i in range(1, 9)])
                labelsize = 13 if not poster_plot else 20
                ax.tick_params(axis='both', which='major', labelsize=labelsize)
                legend_fontsize = 18
                ax.set_ylabel(EUROCODE_RETURN_LEVEL_STR, fontsize = legend_fontsize)
                ax.set_xlabel('Altitude (m)', fontsize = legend_fontsize)
                ax.set_ylim([0.0, 12.0])
                ax.set_yticks(list(range(0, 13, 2)))
                ax.set_xlim([0, 2000])
                ax.set_xticks(list(range(0, 2001, 500)))
                prop = {'size': 20} if poster_plot else {}
                ax.legend(prop=prop, loc='upper left')

                ax.grid()
                plt.show()


def main_eurocode_map(ax=None):
    if ax is None:
        ax = plt.gca()
        massif_name_to_color = {m: r.eurocode_color for m, r in massif_name_to_eurocode_region.items()}
        AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color, scaled=True,
                                      axis_off=True)


if __name__ == '__main__':
    main_eurocode_norms(poster_plot=True)
    # main_eurocode_map()
