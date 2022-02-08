import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color, gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str, AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp
from extreme_fit.distribution.gev.gev_params import GevParams
from projects.projected_extreme_snowfall.results.part_2.average_bias import compute_average_bias, plot_bias_repartition
from projects.projected_extreme_snowfall.results.part_3.plot_gcm_rcm_effects import load_total_effect


def plot_bias_reduction(gcm_rcm_couple_to_study, massif_name, reference_study, visualizer, scenario):
    average_bias, gcm_rcm_couple_to_biases = compute_average_bias(gcm_rcm_couple_to_study, massif_name, reference_study,
                                                                  show=False)

    ax = plt.gca()

    # Plot bias for the GCM
    ax2 = ax.twinx()

    for gcm, color in gcm_to_color.items():
        margin_function = visualizer.massif_name_to_one_fold_fit[massif_name].best_margin_function_from_fit
        temp = year_to_global_mean_temp(gcm, scenario)[1959]
        gev_params_obs = margin_function.get_params(np.array([temp])) # type: GevParams
        gev_params_shifted = margin_function.get_params(np.array([temp, gcm])) # type: GevParams

        shift_mean = 100 * (gev_params_shifted.mean - gev_params_obs.mean) / gev_params_obs.mean
        shift_std = 100 * (gev_params_shifted.std - gev_params_obs.std) / gev_params_obs.std
        ax2.scatter([shift_mean], [shift_std], color=color, marker='^', label=gcm)
    ax2.legend(prop={'size': 7}, loc='upper left')

    # Plot bias for the ensemble member
    for gcm_rcm_couple, biases in gcm_rcm_couple_to_biases.items():
        xi, yi = biases
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([xi], [yi], color=color, marker='o', label=name)

    # Plot average bias
    plot_bias_repartition(average_bias, ax, 'SAFRAN')
    visualizer.plot_name = 'plot bias repartition'
    visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()
