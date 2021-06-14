from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.distribution.gev.gev_params import GevParams


def plot_average_bias(gcm_rcm_couple_to_study, massif_name, average_bias_obs, alpha):
    absolute_average_bias_obs = average_bias_obs
    gcm_rcm_couple_to_absolute_average_bias = {}
    for gcm_rcm_couple in gcm_rcm_couple_to_study.keys():
        gcm_rcm_couple_to_study_fake = gcm_rcm_couple_to_study.copy()
        reference_study = gcm_rcm_couple_to_study_fake.pop(gcm_rcm_couple)
        _, _, average_bias = compute_average_bias(gcm_rcm_couple_to_study_fake, massif_name, reference_study)
        gcm_rcm_couple_to_absolute_average_bias[gcm_rcm_couple] = average_bias
    ax = plt.gca()
    ax.scatter([absolute_average_bias_obs[0]], [absolute_average_bias_obs[1]], color="k", marker='x', label="SAFRAN")

    for gcm_rcm_couple, absolute_average_bias in gcm_rcm_couple_to_absolute_average_bias.items():
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([absolute_average_bias[0]], [absolute_average_bias[1]], color=color, marker='x', label=name)

    alpha_location = alpha
    alpha_scale = alpha
    lim_left, lim_right = ax.get_ylim()
    ax.vlines(0, ymin=lim_left, ymax=lim_right)
    for shift in [-1, 1]:
        ax.vlines(absolute_average_bias_obs[0] + shift * alpha_location, ymin=lim_left, ymax=lim_right, linestyle='--')
    lim_left, lim_right = ax.get_xlim()
    for shift in [-1, 1]:
        ax.hlines(absolute_average_bias_obs[1] + shift * alpha_scale, xmin=lim_left, xmax=lim_right, linestyle='--')
    ax.hlines(0, xmin=lim_left, xmax=lim_right)

    ax.legend()
    ax.set_xlabel('Average relative bias for the location parameter')
    ax.set_ylabel('Average relative bias for the scale parameter')

    # plt.show()
    plt.close()
    gcm_rcm_couples_selected = []
    for gcm_rcm_couple, absolute_average_bias in gcm_rcm_couple_to_absolute_average_bias.items():
        if np.abs(absolute_average_bias[0] - absolute_average_bias_obs[0]) < alpha_location:
            if np.abs(absolute_average_bias[1] - absolute_average_bias_obs[1]) < alpha_scale:
                gcm_rcm_couples_selected.append(gcm_rcm_couple)
    return gcm_rcm_couples_selected


def plot_bias(gcm_rcm_couple_to_study, massif_name, safran_study):
    # Create plot
    biases_matrix, gcm_rcm_couple_to_biases, average_bias = compute_average_bias(gcm_rcm_couple_to_study, massif_name,
                                                                                 safran_study)
    # max_bias = np.max(biases_matrix, axis=0)
    # min_bias = np.min(biases_matrix, axis=0)
    # print(min_bias, average_bias, max_bias)
    ax = plt.gca()
    for gcm_rcm_couple, biases in gcm_rcm_couple_to_biases.items():
        xi, yi = biases
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([xi], [yi], color=color, marker='o', label=name)
    xi, yi = average_bias
    ax.scatter([xi], [yi], color="k", marker='x', label="Average")
    lim_left, lim_right = ax.get_xlim()
    ax.hlines(0, xmin=lim_left, xmax=lim_right)
    lim_left, lim_right = ax.get_ylim()
    ax.vlines(0, ymin=lim_left, ymax=lim_right)
    ax.legend()
    ax.set_xlabel('Relative bias for the location parameter')
    ax.set_ylabel('Relative bias for the scale parameter')
    # plt.show()
    plt.close()
    return average_bias


def compute_average_bias(gcm_rcm_couple_to_study, massif_name, safran_study):
    gcm_rcm_couple_to_biases = OrderedDict()
    for gcm_rcm_couple, study_for_comparison in gcm_rcm_couple_to_study.items():
        biases = compute_bias(massif_name, safran_study, study_for_comparison, relative=True)
        gcm_rcm_couple_to_biases[gcm_rcm_couple] = biases
    biases_matrix = np.array(list(gcm_rcm_couple_to_biases.values()))
    average_bias = np.mean(biases_matrix, axis=0)
    return biases_matrix, gcm_rcm_couple_to_biases, average_bias


def compute_bias(massif_name, study_reference: AbstractStudy,
                 study_for_comparison: AbstractAdamontStudy,
                 relative
                 ):
    start_1, end_1 = study_reference.start_year_and_stop_year
    start_2, end_2 = study_for_comparison.start_year_and_stop_year
    assert start_1 == start_2
    assert end_1 == end_2
    gev_param1 = study_reference.massif_name_to_stationary_gev_params[massif_name]
    gev_param2 = study_for_comparison.massif_name_to_stationary_gev_params[massif_name]
    biases = []
    for param_name in GevParams.PARAM_NAMES[:2]:
        param2, param1 = gev_param2.to_dict()[param_name], gev_param1.to_dict()[param_name]
        bias = param2 - param1
        if relative:
            bias *= 100 / param1
        biases.append(bias)
    return np.array(biases)

def load_study(altitudes, gcm_rcm_couples, safran_study_class, scenario, study_class):
    altitude = altitudes[0]
    year_min, year_max = 1959, 2019
    safran_study = safran_study_class(altitude=altitude, year_min=year_min, year_max=year_max)
    gcm_rcm_couple_to_study = OrderedDict()
    for gcm_rcm_couple in gcm_rcm_couples:
        gcm, rcm = gcm_rcm_couple
        if (gcm == 'HadGEM2-ES') or (rcm == 'RCA4'):
            continue
        study = study_class(altitude=altitude, scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                            year_min=year_min, year_max=year_max)
        gcm_rcm_couple_to_study[gcm_rcm_couple] = study
    return gcm_rcm_couple_to_study, safran_study

