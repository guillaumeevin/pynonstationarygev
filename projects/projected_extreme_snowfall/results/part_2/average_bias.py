from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_rcm_couple_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import gcm_rcm_couple_to_str
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer


def plot_average_bias(gcm_rcm_couple_to_study, massif_name, average_bias_obs, alpha, show=False):
    gcm_rcm_couple_to_average_bias = {}
    gcm_rcm_couple_to_gcm_rcm_couple_to_biases = {}
    for gcm_rcm_couple in gcm_rcm_couple_to_study.keys():
        gcm_rcm_couple_to_study_fake = gcm_rcm_couple_to_study.copy()
        reference_study = gcm_rcm_couple_to_study_fake.pop(gcm_rcm_couple)
        if reference_study.start_year_and_stop_year[0] == 1959:
            average_bias, gcm_rcm_couple_to_biases = compute_average_bias(gcm_rcm_couple_to_study_fake, massif_name, reference_study, show=False)
            gcm_rcm_couple_to_average_bias[gcm_rcm_couple] = average_bias
            gcm_rcm_couple_to_gcm_rcm_couple_to_biases[gcm_rcm_couple] = gcm_rcm_couple_to_biases
    ax = plt.gca()

    circle = plt.Circle(tuple(average_bias_obs), alpha, color='k', fill=False, linestyle='--')
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
    ax.set_xlabel('Average relative bias for the mean of annual maxima (\%)')
    ax.set_ylabel('Average relative bias for the std of annual maxima (\%)')

    if show in [None, True]:
        save_to_file = True if show is None else False
        any_study = list(gcm_rcm_couple_to_study.values())[0]
        visualizer = StudyVisualizer(any_study, save_to_file=save_to_file)
        visualizer.plot_name = 'plot average bias repartition for {}'.format(massif_name)
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()
    gcm_rcm_couples_selected = []
    for gcm_rcm_couple, average_bias in gcm_rcm_couple_to_average_bias.items():
        difference_average_bias = average_bias - average_bias_obs
        if np.linalg.norm(difference_average_bias) <= alpha:
            gcm_rcm_couples_selected.append(gcm_rcm_couple)
    return gcm_rcm_couples_selected, gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases


def compute_average_bias(gcm_rcm_couple_to_study, massif_name, reference_study, show=True):
    # Compute average biais & bias
    gcm_rcm_couple_to_biases = OrderedDict()
    for gcm_rcm_couple, study_for_comparison in gcm_rcm_couple_to_study.items():
        biases = compute_bias(massif_name, reference_study, study_for_comparison, True)
        gcm_rcm_couple_to_biases[gcm_rcm_couple] = biases
    biases_matrix = np.array(list(gcm_rcm_couple_to_biases.values()))
    average_bias = np.mean(biases_matrix, axis=0)
    # Plot the bias
    plot_bias(reference_study, average_bias, gcm_rcm_couple_to_biases, massif_name, show)
    return average_bias, gcm_rcm_couple_to_biases


def plot_bias(study_reference, average_bias, gcm_rcm_couple_to_biases, massif_name, show):
    ax = plt.gca()

    for gcm_rcm_couple, biases in gcm_rcm_couple_to_biases.items():
        xi, yi = biases
        color = gcm_rcm_couple_to_color[gcm_rcm_couple]
        name = gcm_rcm_couple_to_str(gcm_rcm_couple)
        ax.scatter([xi], [yi], color=color, marker='o', label=name)

    name = gcm_rcm_couple_to_str(study_reference.gcm_rcm_couple) \
        if isinstance(study_reference, AbstractAdamontStudy) else 'SAFRAN'
    plot_bias_repartition(average_bias, ax, name)
    if show in [None, True]:
        save_to_file = True if show is None else False
        visualizer = StudyVisualizer(study_reference, save_to_file=save_to_file)
        visualizer.plot_name = 'plot bias repartition for {}'.format(massif_name)
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()


def plot_time_series(massif_name, study_reference, gcm_rcm_couple_to_study, show):
    ax = plt.gca()
    # Plot the ensemble member
    linewidth = 2
    for gcm_rcm_couple, color in list(gcm_rcm_couple_to_color.items())[::-1]:
        if gcm_rcm_couple in gcm_rcm_couple_to_study:
            study = gcm_rcm_couple_to_study[gcm_rcm_couple]
            if massif_name in study.massif_name_to_annual_maxima:
                x = study.ordered_years
                y = study.massif_name_to_annual_maxima[massif_name]
                label = gcm_rcm_couple_to_str(gcm_rcm_couple)
                color = gcm_rcm_couple_to_color[gcm_rcm_couple]
                ax.plot(x, y, linewidth=linewidth, label=label, color=color)

    # Plot the pseudo observation on top
    x = study_reference.ordered_years
    y = study_reference.massif_name_to_annual_maxima[massif_name]
    label = 'Pseudo observation'
    color = 'black'
    ax.plot(x, y, linewidth=linewidth * 2, label=label, color=color)
    ax.set_ylabel(
        'Annual maxima with {} as pseudo observations'.format(gcm_rcm_couple_to_str(study_reference.gcm_rcm_couple)))
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], prop={'size': 6}, ncol=3)

    ax.set_xlabel('Years')
    plt.tick_params(axis='both', which='major', labelsize=7)
    ax.set_xlim((1950, 2100))

    if show in [None, True]:
        save_to_file = True if show is None else False
        visualizer = StudyVisualizer(study_reference, save_to_file=save_to_file, show=show)
        visualizer.plot_name = 'time series'
        visualizer.show_or_save_to_file(add_classic_title=False, no_title=True)
    plt.close()


def plot_bias_repartition(average_bias, ax, name, skip_percent=True):
    percent = '\%' if skip_percent else '%'
    xi, yi = average_bias
    ax.scatter([xi], [yi], color="k", marker='x', label="Average bias")
    ax.set_xlim(-25, 45)
    ax.set_ylim(-25, 45)
    lim_left, lim_right = ax.get_xlim()
    ax.hlines(0, xmin=lim_left, xmax=23)
    lim_left, lim_right = ax.get_ylim()
    ax.vlines(0, ymin=lim_left, ymax=lim_right)
    ax.legend(prop={'size': 7}, loc='lower right', ncol=1)
    common_label = 'Bias w.r.t {}'.format(name)
    common_label += ' for the {} (' + percent + ')'
    ax.set_xlabel(common_label.format('mean'))
    ax.set_ylabel(common_label.format('standard deviation'))


def compute_bias(massif_name, study_reference: AbstractStudy,
                 study_for_comparison: AbstractAdamontStudy,
                 relative,
                 ):
    ordered_years = set(study_reference.ordered_years).intersection(set(study_for_comparison.ordered_years))
    annual_maxima1 = [m for year, m in study_reference.year_to_annual_maxima_for_a_massif(massif_name).items()
                      if year in ordered_years]
    annual_maxima2 = [m for year, m in study_for_comparison.year_to_annual_maxima_for_a_massif(massif_name).items()
                      if year in ordered_years]
    assert len(annual_maxima1) <= 61, "bias should be at maximum computed on the 1959-2019 period"
    assert len(annual_maxima2) <= 61, "bias should be at maximum computed on the 1959-2019 period"
    assert len(annual_maxima1) == len(annual_maxima2), "{} vs {}".format(len(annual_maxima1), len(annual_maxima2))
    biases = []
    for f in [np.mean, np.std]:
        moment_ref, moment_comparison = [f(maxima) for maxima in [annual_maxima1, annual_maxima2]]
        bias = moment_comparison - moment_ref
        if relative:
            bias *= 100 / moment_ref
        biases.append(bias)
    return np.array(biases)


def load_study(altitude, gcm_rcm_couples, safran_study_class, scenario, study_class):
    year_min, year_max = 1959, 2019
    safran_study = safran_study_class(altitude=altitude, year_min=year_min, year_max=year_max)
    gcm_rcm_couple_to_study = OrderedDict()
    for gcm_rcm_couple in gcm_rcm_couples:
        study = study_class(altitude=altitude, scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                            year_min=year_min, year_max=year_max)
        gcm_rcm_couple_to_study[gcm_rcm_couple] = study
    return gcm_rcm_couple_to_study, safran_study

