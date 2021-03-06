from typing import List

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color, gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projected_extremes.section_results.utils.plot_relative_change_in_return_level import set_plot_name


def plot_gcm_rcm_effects(massif_names, visualizer_list: List[
    AltitudesStudiesVisualizerForNonStationaryModels], climate_coordinates_names_with_param_effects_to_extract,
                         safran_study_class,
                         gcm_rcm_couples, param_name,
                         ):
    ax = plt.gca()
    altitudes = [v.study.altitude for v in visualizer_list]
    visualizer = visualizer_list[0]
    assert len(massif_names) == 1
    assert climate_coordinates_names_with_param_effects_to_extract is not None
    massif_name = massif_names[0]
    all_effects = []
    for gcm_rcm_couple in gcm_rcm_couples:
        effects = plot_curve_gcm_rcm_effect(ax, massif_name, visualizer_list, gcm_rcm_couple, param_name)
        all_effects.append(effects)
    all_effects = np.array(all_effects)
    mean_effects = np.mean(all_effects, axis=0)
    assert len(mean_effects) == len(altitudes)

    ax.plot(mean_effects, altitudes, label='Mean effect', color='k', linewidth=4)

    effect_name = '-'.join([c.replace('coord_', '').upper() for c in climate_coordinates_names_with_param_effects_to_extract])
    param_name_str = GevParams.full_name_from_param_name(param_name)
    xlabel = '{} effect for the {} parameter'.format(effect_name, param_name_str)
    ax.vlines(0, ymin=altitudes[0], ymax=altitudes[-1], color='k', linestyles='dashed')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Altitude (m)')
    title = '{} parameter'.format(param_name_str)
    ax.set_ylim(top=altitudes[-1] + 700)
    ax.yaxis.set_ticks(altitudes)
    size = 7 if len(climate_coordinates_names_with_param_effects_to_extract) == 2 else 10
    ax.legend(ncol=3, prop={'size': size})
    set_plot_name(climate_coordinates_names_with_param_effects_to_extract, safran_study_class, title, visualizer,
                  massif_name)
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)

    plt.close()


def plot_curve_gcm_rcm_effect(ax, massif_name, visualizer_list: List[AltitudesStudiesVisualizerForNonStationaryModels],
                              gcm_rcm_couple, param_name):
    altitudes = [v.study.altitude for v in visualizer_list]
    effects = []
    for visualizer in visualizer_list[:]:
        total_effect = load_total_effect(gcm_rcm_couple, massif_name, param_name, visualizer)
        effects.append(total_effect)
    if len(gcm_rcm_couple) == 2:
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        label = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.plot(effects, altitudes, label=label, color=color)
    else:
        is_gcm = gcm_rcm_couple[0] in gcm_to_color
        if is_gcm:
            gcm = gcm_rcm_couple[0]
            ax.plot(effects, altitudes, label=gcm, color=gcm_to_color[gcm])
        else:
            rcm = gcm_rcm_couple[0]
            ax.plot(effects, altitudes, label=rcm)
    return effects


def load_total_effect(gcm_rcm_couple, massif_name, param_name, visualizer, temporal_coordinate=1.0):
    one_fold_fit = visualizer.massif_name_to_one_fold_fit[massif_name]
    margin_function = one_fold_fit.best_margin_function_from_fit
    full_climate_coordinate = gcm_rcm_couple
    total_effect = margin_function.load_total_effect_for_gcm_rcm_couple(full_climate_coordinate, param_name,
                                                                        temporal_coordinate)
    return total_effect
