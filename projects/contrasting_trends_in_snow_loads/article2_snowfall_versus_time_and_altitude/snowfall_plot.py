from typing import Dict
import matplotlib.pyplot as plt

import numpy as np
from sklearn.linear_model import LinearRegression

from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.plot_utils import plot_against_altitude
from projects.contrasting_trends_in_snow_loads.article2_snowfall_versus_time_and_altitude.study_visualizer_for_mean_values import \
    StudyVisualizerForMeanValues


def fit_linear_regression(x, y):
    X = np.array(x).reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    r2_score = reg.score(X, y)
    a = reg.coef_
    b = reg.intercept_
    return a, b, r2_score


def plot_snowfall_change_mean(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    visualizer = list(altitude_to_visualizer.values())[0]
    study = visualizer.study

    variables = ['mean annual maxima', '50 year return level']
    mean_indicators = [True, False]
    l = list(zip(variables, mean_indicators))
    for variable, mean_indicator in l[:1]:
        for relative in [False, True]:
            if relative:
                variable += str(' (relative)')
            # Plot the curve for the evolution of the mean
            massif_name_to_a, massif_name_to_b, massif_name_to_r2_score = plot_mean(altitude_to_visualizer,
                                                                                    derivative=True, mean_indicator=mean_indicator,
                                                                                    relative=relative)
            # Augmentation every km
            massif_name_to_augmentation_every_km = {m: a * 1000 for m, a in massif_name_to_a.items()}
            massif_to_r2_score_text = {k: str(round(v, 2)) for k, v in massif_name_to_r2_score.items()}

            visualizer.plot_abstract_fast(massif_name_to_augmentation_every_km,
                                          label='Slope for changes of {} of {}\n for every km of elevation ({})'.format(
                                              variable, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)], study.variable_unit),
                                          add_x_label=False,
                                          add_text=True, massif_name_to_text=massif_to_r2_score_text)
            # Value at 2000 m
            massif_name_to_mean_at_2000 = {m: a * 2000 + massif_name_to_b[m] for m, a in massif_name_to_a.items()}
            visualizer.plot_abstract_fast(massif_name_to_mean_at_2000,
                                          label='Changes of {} \nof {} at 2000 m ({})'.format(
                                              variable, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)], study.variable_unit),
                                          add_x_label=False,
                                          add_text=True, massif_name_to_text=massif_to_r2_score_text)
            # Altitude for the change of dynamic
            massif_name_to_altitude_change_dynamic = {m: - massif_name_to_b[m] / a for m, a in massif_name_to_a.items()}
            # Keep only those that are in a reasonable range
            massif_name_to_altitude_change_dynamic = {m: d for m, d in massif_name_to_altitude_change_dynamic.items()
                                                      if 0 < d < 3000}
            visualizer.plot_abstract_fast(massif_name_to_altitude_change_dynamic,
                                          label=('Altitude when the changes equal zero for %s (m)' % variable),
                                          add_x_label=False, graduation=500,
                                          add_text=True, massif_name_to_text=massif_to_r2_score_text)


def plot_snowfall_mean(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues]):
    visualizer = list(altitude_to_visualizer.values())[0]
    study = visualizer.study

    variables = ['mean annual maxima', '50 year return level']
    mean_indicators = [True, False]
    for variable, mean_indicator in zip(variables, mean_indicators):
        # Plot the curve for the evolution of the mean
        massif_name_to_a, massif_name_to_b, massif_name_to_r2_score = plot_mean(altitude_to_visualizer,
                                                                                derivative=False,
                                                                                mean_indicator=mean_indicator)
        # Compute values of interest
        massif_name_to_mean_at_2000 = {m: a * 2000 + massif_name_to_b[m] for m, a in massif_name_to_a.items()}
        massif_name_to_augmentation_every_km = {m: a * 1000 for m, a in massif_name_to_a.items()}
        massif_name_to_relative_augmentation_between_2000_and_3000_every_km = {
            m: 100 * (a * 1000) / (a * 2000 + massif_name_to_b[m]) for m, a in massif_name_to_a.items()}
        massif_name_to_relative_augmentation_between_1000_and_2000_every_km = {
            m: 100 * (a * 1000) / (a * 1000 + massif_name_to_b[m]) for m, a in massif_name_to_a.items()}
        massif_to_r2_score_text = {k: str(round(v, 2)) for k, v in massif_name_to_r2_score.items()}
        # Augmentation every km
        ds = [massif_name_to_augmentation_every_km, massif_name_to_relative_augmentation_between_1000_and_2000_every_km,
              massif_name_to_relative_augmentation_between_2000_and_3000_every_km]
        prefixs = ['Augmentation', 'Relative augmentation between 1000m and 2000m',
                   'Relative augmentation between 2000m and 3000m']
        graduations = [1.0, 10.0, 10.0]
        for d, prefix, graduation in zip(ds, prefixs, graduations):
            visualizer.plot_abstract_fast(d,
                                          graduation=10.0,
                                          label=prefix + ' of {} of {} \nfor every km of elevation ({})'.format(
                                              variable, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)],
                                              study.variable_unit),
                                          add_x_label=False, negative_and_positive_values=False,
                                          add_text=True,
                                          massif_name_to_text=massif_to_r2_score_text
                                          )
        # Value at 2000 m
        visualizer.plot_abstract_fast(massif_name_to_mean_at_2000, label='{} of {} at 2000 m ()'.format(
            variable, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)], study.variable_unit),
                                      add_x_label=False, negative_and_positive_values=False,
                                      add_text=True, massif_name_to_text=massif_to_r2_score_text)


def plot_mean(altitude_to_visualizer: Dict[int, StudyVisualizerForMeanValues], derivative=False, mean_indicator=True,
              relative=False):
    ax = plt.gca()
    massif_name_to_linear_regression_result = {}

    visualizers = list(altitude_to_visualizer.values())
    visualizer = visualizers[0]
    study = visualizer.study

    for massif_id, massif_name in enumerate(visualizer.study.all_massif_names()):
        altitudes_massif = [a for a, v in altitude_to_visualizer.items()
                            if massif_name in v.massif_name_to_trend_test_that_minimized_aic]
        if len(altitudes_massif) >= 2:
            trend_tests = [altitude_to_visualizer[a].massif_name_to_trend_test_that_minimized_aic[massif_name]
                           for a in altitudes_massif]
            nb_years = 50
            return_period = 50
            if mean_indicator:
                indicator_str = 'mean'
            else:
                indicator_str = '50 year return level'
            res = [(a, t) for i, (a, t) in enumerate(zip(altitudes_massif, trend_tests))]
            if derivative:
                if mean_indicator:
                    if relative:
                        res = [(a, t.relative_change_in_mean_for_the_last_x_years(nb_years=nb_years)) for a, t in res]
                    else:
                        res = [(a, t.change_in_mean_for_the_last_x_years(nb_years=nb_years)) for a, t in res]
                else:
                    if relative:
                        res = [(a, t.relative_change_in_50_year_return_level_for_the_last_x_years(nb_years=nb_years, return_period=return_period)) for a, t in res]
                    else:
                        res = [(a, t.change_in_50_year_return_level_for_the_last_x_years(nb_years=nb_years, return_period=return_period)) for a, t in res]
                if len(res) > 0:
                    altitudes_values, values = zip(*res)
                else:
                    altitudes_values, values = [], []
                indicator_str = 'Change of the {} for the last {} years  \nfor non-stationary models'.format(indicator_str, nb_years)
                if relative:
                    indicator_str += ' (relative)'
            else:
                if mean_indicator:
                    values = [t.unconstrained_estimator_gev_params_last_year.mean for t in trend_tests]
                else:
                    values = [t.unconstrained_estimator_gev_params_last_year.return_level(return_period=return_period)
                              for t in trend_tests]
                altitudes_values = altitudes_massif
            # Plot
            if len(altitudes_values) >= 2:
                massif_name_to_linear_regression_result[massif_name] = fit_linear_regression(altitudes_values, values)
                plot_values_against_altitudes(ax, altitudes_values, massif_id, massif_name, indicator_str, study, values,
                                              visualizer)
    ax.legend(prop={'size': 7}, ncol=3)
    visualizer.show_or_save_to_file(dpi=500, add_classic_title=False)
    plt.close()

    return [{m: t[i][0] if i == 0 else t[i]
             for m, t in massif_name_to_linear_regression_result.items()} for i in range(3)]


def plot_values_against_altitudes(ax, altitudes, massif_id, massif_name, moment, study, values, visualizer):
    plot_against_altitude(x_ticks=altitudes, ax=ax, massif_id=massif_id, massif_name=massif_name, values=values)
    plot_name = '{} annual maxima of {}'.format(moment, SCM_STUDY_CLASS_TO_ABBREVIATION[type(study)])
    ax.set_ylabel('{} ({})'.format(plot_name, study.variable_unit), fontsize=15)
    ax.set_xlabel('altitudes', fontsize=15)
    # lim_down, lim_up = ax.get_ylim()
    # lim_up += (lim_up - lim_down) / 3
    # ax.set_ylim([lim_down, lim_up])
    ax.tick_params(axis='both', which='major', labelsize=13)
    visualizer.plot_name = plot_name
