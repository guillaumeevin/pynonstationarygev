import matplotlib.pyplot as plt
from copy import deepcopy, copy
import numpy as np

from experiment.eurocode_data.eurocode_region import C2, E, C1
from experiment.eurocode_data.massif_name_to_departement import massif_name_to_eurocode_region
from experiment.eurocode_data.utils import EUROCODE_RETURN_LEVEL_STR
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from root_utils import get_display_name_from_object_type




def main_eurocode_norms(ax=None):
    if ax is None:
        ax = plt.gca()
        altitudes = np.linspace(200, 2000)
        for region_class in [C1, C2, E][:]:
            region_object = region_class()
            label = get_display_name_from_object_type(region_class) + ' Eurocodes region'
            linestyle = '--' if region_class == C2 else '-'
            region_object.plot_eurocode_snow_load_on_ground_characteristic_value_variable_action(ax, altitudes, label=label, linestyle=linestyle)
            if region_class == E:
                ax.xaxis.set_ticks([250 * i for i in range(1, 9)])
                ax.tick_params(axis='both', which='major', labelsize=13)
                ax.set_ylabel(EUROCODE_RETURN_LEVEL_STR)
                ax.set_xlabel('Altitude (m)')
                ax.set_ylim([0.0, 12.0])
                ax.set_yticks(list(range(0, 13, 2)))
                ax.set_xlim([0, 2000])
                ax.set_xticks(list(range(0, 2001, 500)))
                handles, labels = ax.get_legend_handles_labels()
                # Copy the handle to not modify the original plot
                handle1 = copy(handles[1])
                handle1.set_linestyle('-')
                handles[1] = handle1
                ax.legend(handles, labels)
                ax.grid()
                plt.show()


def main_eurocode_map(ax=None):
    if ax is None:
        ax = plt.gca()
        massif_name_to_color = {m: r.eurocode_color for m, r in massif_name_to_eurocode_region.items()}
        AbstractStudy.visualize_study(ax, massif_name_to_color=massif_name_to_color, scaled=True)


if __name__ == '__main__':
    main_eurocode_norms()
    # main_eurocode_map()
