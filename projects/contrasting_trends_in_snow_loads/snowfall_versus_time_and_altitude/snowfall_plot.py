from typing import Dict
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression

from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from projects.contrasting_trends_in_snow_loads.snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues


def fit_linear_regression(x, y):
    X = np.array(x).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    a = reg.coef_
    b = reg.intercept_
    return a, b, r2_score


def plot_snowfall_mean(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    visualizer = list(altitude_to_visualizer.values())[0]
    # Plot the curve for the evolution of the mean
    massif_name_to_a, massif_name_to_b, massif_name_to_r2_score = plot_mean(altitude_to_visualizer, derivative=False)
    # Plot map with the coefficient a
    visualizer.plot_abstract_fast(massif_name_to_a, label='a')
    visualizer.plot_abstract_fast(massif_name_to_b, label='b')
    visualizer.plot_abstract_fast(massif_name_to_r2_score, label='r2')


def plot_mean(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues], derivative=False):
    massif_name_to_linear_regression_result = {}

    altitudes = list(altitude_to_visualizer.keys())
    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    study = visualizer.study
    year = study.year_max

    for massif_id, massif_name in enumerate(visualizer.study.all_massif_names()):
        altitudes_massif = [a for a, v in altitude_to_visualizer.items()
                            if massif_name in v.massif_name_to_trend_test_that_minimized_aic]
        if len(altitudes_massif) >= 2:
            trend_tests = [altitude_to_visualizer[a].massif_name_to_trend_test_that_minimized_aic[massif_name]
                           for a in altitudes_massif]
            if derivative:
                moment = 'time derivative of the mean'
                values = [t.first_derivative_mean_value(year=year) for t in trend_tests]
            else:
                moment = 'mean'
                values = [t.mean_value(year=year) for t in trend_tests]
            massif_name_to_linear_regression_result[massif_name] = fit_linear_regression(altitudes, values)
            plot_values_against_altitudes(altitudes_massif, massif_id, massif_name, moment, study, values, visualizer)

    return [{m: t[i][0] if i == 0 else t[i]
             for m, t in massif_name_to_linear_regression_result.items()} for i in range(3)]


def plot_values_against_altitudes(altitudes, massif_id, massif_name, moment, study, values, visualizer):
    ax = plt.gca()
    plot_against_altitude(altitudes=altitudes, ax=ax, massif_id=massif_id, massif_name=massif_name, values=values)
    plot_name = '{} annual maxima of {}'.format(moment, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)])
    ax.set_ylabel('{} ({})'.format(plot_name, study.variable_unit), fontsize=15)
    ax.set_xlabel('altitudes', fontsize=15)
    lim_down, lim_up = ax.get_ylim()
    lim_up += (lim_up - lim_down) / 3
    ax.set_ylim([lim_down, lim_up])
    ax.tick_params(axis='both', which='major', labelsize=13)
    visualizer.plot_name = plot_name
    visualizer.show_or_save_to_file(dpi=500, add_classic_title=False)
