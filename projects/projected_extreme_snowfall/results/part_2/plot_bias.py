from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import fitted_stationary_gev


def plot_average_bias(gcm_rcm_couple_to_study, massif_name, average_bias_obs, alpha, show=False):
    average_bias_obs = average_bias_obs
    gcm_rcm_couple_to_average_bias = {}
    for gcm_rcm_couple in gcm_rcm_couple_to_study.keys():
        gcm_rcm_couple_to_study_fake = gcm_rcm_couple_to_study.copy()
        reference_study = gcm_rcm_couple_to_study_fake.pop(gcm_rcm_couple)
        if reference_study.start_year_and_stop_year[0] == 1959:
            _, _, average_bias = compute_average_bias(gcm_rcm_couple_to_study_fake, massif_name, reference_study)
            gcm_rcm_couple_to_average_bias[gcm_rcm_couple] = average_bias
    ax = plt.gca()

    norm_obs = np.linalg.norm(average_bias_obs)
    for shift, color in [(alpha, 'k'), (-alpha, 'k')]:
        obs_shift = max(0, norm_obs + shift)
        circle = plt.Circle((0, 0), obs_shift, color=color, fill=False, linestyle='--')
        ax.add_patch(circle)

    ax.scatter([average_bias_obs[0]], [average_bias_obs[1]], color="k", marker='x', label="SAFRAN")

    for gcm_rcm_couple, average_bias in gcm_rcm_couple_to_average_bias.items():
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([average_bias[0]], [average_bias[1]], color=color, marker='x', label=name)



    lim_left, lim_right = ax.get_ylim()
    ax.vlines(0, ymin=lim_left, ymax=lim_right)
    lim_left, lim_right = ax.get_xlim()
    ax.hlines(0, xmin=lim_left, xmax=lim_right)

    ax.legend(prop={'size': 7})
    ax.set_xlabel('Average relative bias for the mean of annual maxima for 1959-2019 (\%)')
    ax.set_ylabel('Average relative bias for the std of annual maxima for 1959-2019 (\%)')

    if show:
        plt.show()
    plt.close()
    gcm_rcm_couples_selected = []
    for gcm_rcm_couple, average_bias in gcm_rcm_couple_to_average_bias.items():
        norm_gcm_rcm_couple = np.linalg.norm(average_bias)
        if np.abs(norm_gcm_rcm_couple - norm_obs) <= alpha:
            gcm_rcm_couples_selected.append(gcm_rcm_couple)
    return gcm_rcm_couples_selected


def plot_bias(gcm_rcm_couple_to_study, massif_name, safran_study,
              gcm_rcm_couple_to_params_effects=None,
              show=False):
    # Create plot
    biases_matrix, gcm_rcm_couple_to_biases, average_bias = compute_average_bias(gcm_rcm_couple_to_study, massif_name,
                                                                                 safran_study,
                                                                                 gcm_rcm_couple_to_params_effects)
    ax = plt.gca()
    xi, yi = average_bias
    ax.scatter([xi], [yi], color="k", marker='x', label="Average relative bias")
    for gcm_rcm_couple, biases in gcm_rcm_couple_to_biases.items():
        xi, yi = biases
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([xi], [yi], color=color, marker='o', label=name)

    lim_left, lim_right = ax.get_xlim()
    ax.hlines(0, xmin=lim_left, xmax=lim_right)
    lim_left, lim_right = ax.get_ylim()
    ax.vlines(0, ymin=lim_left, ymax=lim_right)
    ax.legend(prop={'size': 7}, loc='lower right', ncol=1)
    ax.set_xlabel('Relative bias for the mean of annual maxima for 1959-2019 (\%)')
    ax.set_ylabel('Relative bias for the standard deviation of annual maxima for 1959-2019 (\%)')
    if show:
        plt.show()
    plt.close()
    return average_bias


def compute_average_bias(gcm_rcm_couple_to_study, massif_name, reference_study,
                         gcm_rcm_couple_to_params_effects=None):
    gcm_rcm_couple_to_biases = OrderedDict()
    for gcm_rcm_couple, study_for_comparison in gcm_rcm_couple_to_study.items():
        if gcm_rcm_couple_to_params_effects is not None:
            params_effects = gcm_rcm_couple_to_params_effects[gcm_rcm_couple]
        else:
            params_effects = None
        biases = compute_bias(massif_name, reference_study, study_for_comparison, True,
                              params_effects)
        gcm_rcm_couple_to_biases[gcm_rcm_couple] = biases
    biases_matrix = np.array(list(gcm_rcm_couple_to_biases.values()))
    average_bias = np.mean(biases_matrix, axis=0)
    return biases_matrix, gcm_rcm_couple_to_biases, average_bias


def compute_bias(massif_name, study_reference: AbstractStudy,
                 study_for_comparison: AbstractAdamontStudy,
                 relative,
                params_effects=None
                 ):
    gev_param_reference, gev_param_comparison = load_gev_params(massif_name, study_for_comparison, study_reference)
    if params_effects is not None:
        gev_param_comparison.location -= params_effects[0]
        log_scale = np.log(gev_param_comparison.scale) - params_effects[1]
        gev_param_comparison.scale = np.exp(log_scale)
        gev_param_comparison.shape -= params_effects[2]
    biases = []
    for f in ['mean', 'std']:
        moment_ref, moment_comparison = [gev_params.__getattribute__(f) for gev_params in [gev_param_reference, gev_param_comparison]]
        bias = moment_comparison - moment_ref
        if relative:
            bias *= 100 / moment_ref
        biases.append(bias)
    return np.array(biases)


def load_gev_params(massif_name, study_for_comparison, study_reference):
    start, end = study_reference.start_year_and_stop_year
    assert start == 1959 and end == 2019
    start_2, end_2 = study_for_comparison.start_year_and_stop_year
    if (start_2 == start) and (end_2 == end):
        gev_param1 = study_reference.massif_name_to_stationary_gev_params[massif_name]
    else:
        annual_maxima_reference = set()
        for year, maxima in zip(study_reference.ordered_years, study_reference.massif_name_to_annual_maxima[massif_name]):
            if start_2 <= year <= end_2:
                annual_maxima_reference.add(maxima)
        gev_param1 = fitted_stationary_gev(annual_maxima_reference)
    try:
        gev_param2 = study_for_comparison.massif_name_to_stationary_gev_params[massif_name]
    except KeyError:
        gev_param2 = fitted_stationary_gev(study_for_comparison.massif_name_to_annual_maxima[massif_name])
    return gev_param1, gev_param2


def load_study(altitude, gcm_rcm_couples, safran_study_class, scenario, study_class):
    year_min, year_max = 1959, 2019
    safran_study = safran_study_class(altitude=altitude, year_min=year_min, year_max=year_max)
    gcm_rcm_couple_to_study = OrderedDict()
    for gcm_rcm_couple in gcm_rcm_couples:
        study = study_class(altitude=altitude, scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                            year_min=year_min, year_max=year_max)
        gcm_rcm_couple_to_study[gcm_rcm_couple] = study
    return gcm_rcm_couple_to_study, safran_study

