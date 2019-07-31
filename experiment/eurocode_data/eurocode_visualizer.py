from collections import OrderedDict
import matplotlib.pyplot as plt

from experiment.eurocode_data.massif_name_to_departement import massif_name_to_departements
from experiment.eurocode_data.region_eurocode import AbstractRegionType
from utils import get_display_name_from_object_type


def display_region_limit(region_type, altitudes, ordered_massif_name_to_quantiles, ordered_massif_name_to_significances=None,
                         display=True):
    assert isinstance(ordered_massif_name_to_quantiles, OrderedDict)
    assert ordered_massif_name_to_significances is None or isinstance(ordered_massif_name_to_significances, OrderedDict)
    # First, select massif name correspond to the region
    massif_name_belong_to_the_region = []
    for massif_name in ordered_massif_name_to_quantiles.keys():
        if any([isinstance(dep.region, region_type) for dep in massif_name_to_departements[massif_name]]):
            massif_name_belong_to_the_region.append(massif_name)
    region_object = region_type() # type: AbstractRegionType
    # Then, display the limit for the region
    fig, ax = plt.subplots(1, 1)
    ax.plot(altitudes, [region_object.eurocode_max_loading(altitude) for altitude in altitudes], label='Eurocode limit')
    # Finally, display the massif curve
    for massif_name in massif_name_belong_to_the_region:
        ax.plot(altitudes, ordered_massif_name_to_quantiles[massif_name], label=massif_name)
    ax.set_title('{} Eurocode region'.format(get_display_name_from_object_type(region_type)))
    ax.set_xlabel('Altitude')
    ax.set_ylabel('0.98 quantile (in N $m^-2$)')
    ax.legend()
    if display:
        plt.show()