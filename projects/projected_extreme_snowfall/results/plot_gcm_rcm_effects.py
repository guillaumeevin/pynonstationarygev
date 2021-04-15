
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.projected_extreme_snowfall.results.plot_relative_change_in_return_level import set_plot_name
from root_utils import get_display_name_from_object_type


def plot_gcm_rcm_effects(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], climate_coordinates_for_plot, climate_coordinates_with_effects,
                                          safran_study_class,
gcm_rcm_couples, param_name
                                          ):
    ax = plt.gca()
    altitudes = [v.study.altitude for v in visualizer_list]
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    assert climate_coordinates_with_effects is not None
    massif_name = massif_names[0]
    for gcm_rcm_couple in gcm_rcm_couples:
        plot_curve_gcm_rcm_effect(ax, massif_name, visualizer_list,
                                  climate_coordinates_with_effects, gcm_rcm_couple, param_name)

    effect_name = '-'.join([c.replace('coord_', '').upper() for c in climate_coordinates_for_plot])
    param_name_str = GevParams.full_name_from_param_name(param_name)
    xlabel = '{} effect for the {} parameter'.format(effect_name, param_name_str)
    ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k', linestyles='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Altitude (m)')
    title = '{} parameter'.format(param_name_str)
    ax.set_ylim(top=altitudes[-1] + 1300)
    size = 7 if len(climate_coordinates_for_plot) == 2 else 10
    ax.legend(ncol=3, prop={'size': size})
    set_plot_name(climate_coordinates_for_plot, safran_study_class, title, visualizer)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()



def plot_curve_gcm_rcm_effect(ax, massif_name, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels],
climate_coordinates_with_effects,
               gcm_rcm_couple, param_name):
    altitudes = [v.study.altitude for v in visualizer_list]
    effects = []
    print("\n\n",climate_coordinates_with_effects, gcm_rcm_couple)
    for visualizer in visualizer_list[:]:
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
        indices = one_fold_fit.dataset.coordinates.get_indices_for_effects(climate_coordinates_with_effects, gcm_rcm_couple)
        assert len(indices) <= 2, indices
        ordered_climate_effects = one_fold_fit.best_function_from_fit.param_name_to_ordered_climate_effects[param_name]
        print(ordered_climate_effects, indices)
        sum_effects = sum([ordered_climate_effects[i] for i in indices])
        effects.append(sum_effects)
    if len(gcm_rcm_couple) == 2:
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        label = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.plot(effects, altitudes, label=label, color=color)
    else:
        ax.plot(effects, altitudes, label=gcm_rcm_couple[0])
