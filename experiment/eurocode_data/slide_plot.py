import matplotlib.pyplot as plt
import numpy as np

from experiment.eurocode_data.eurocode_region import C2, E
from root_utils import get_display_name_from_object_type

if __name__ == '__main__':
    ax = plt.gca()
    altitudes = np.linspace(200, 2000)
    for region_class in [C2, E][1:]:
        region_object = region_class()
        region_object.plot_max_loading(ax, altitudes)
        # ax.set_title(get_display_name_from_object_type(region_object) + ' Eurocodes region')
        ax.set_ylabel('50-year return level (kN $m^-2$)')
        ax.set_xlabel('Altitude (m)')
        ax.set_ylim([0.0, 11.0])
        ax.grid()
        plt.show()