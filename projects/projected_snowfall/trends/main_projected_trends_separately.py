import time
from typing import List

import matplotlib as mpl
import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import WrongYearMinOrYearMax
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoad3Days
from projects.altitude_spatial_model.altitudes_fit.plots.plot_coherence_curves import plot_coherence_curves
from projects.altitude_spatial_model.altitudes_fit.plots.plot_histogram_altitude_studies import \
    plot_histogram_all_models_against_altitudes, plot_histogram_all_trends_against_altitudes, \
    plot_shoe_plot_changes_against_altitude

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from projects.altitude_spatial_model.altitudes_fit.utils_altitude_studies_visualizer import load_visualizer_list

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.plot_total_aic import plot_individual_aic

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days, \
    SafranSnowfall5Days, SafranSnowfall7Days, SafranDateFirstSnowfall, SafranPrecipitation1Day, SafranPrecipitation3Days
from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    study_class = AdamontSnowfall
    scenario = AdamontScenario.rcp85_extended
    gcm_rcm_couples = [('CNRM-CM5', 'CCLM4-8-17'), ('CNRM-CM5', 'RCA4')][1:]

    fast = True
    if fast is None:
        massif_names = None
        altitudes_list = altitudes_for_groups[1:2]
    elif fast:
        massif_names = ['Vanoise', 'Haute-Maurienne', 'Vercors'][:1]
        altitudes_list = altitudes_for_groups[1:3]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    # One plot for each massif name
    for massif_name in massif_names:
        print(massif_name)
        main_loop(altitudes_list, [massif_name], gcm_rcm_couples, study_class, scenario)


def main_loop(altitudes_list, massif_names, gcm_rcm_couples, study_class, scenario):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    altitudes_for_return_levels = [altitudes[-2] for altitudes in altitudes_list]
    print(altitudes_for_return_levels)
    season = Season.annual
    first_year_min, first_year_max = 1951, 2010
    nb_years = first_year_max - first_year_min + 1
    temporal_windows = [(first_year_min + i, first_year_max + i) for i in [30 * j for j in range(4)]]
    all_changes_in_return_levels = []
    for gcm_rcm_couple in gcm_rcm_couples:
        print('Inner', gcm_rcm_couple)
        changes_in_return_levels = []
        for temporal_window in temporal_windows:
            year_min, year_max = temporal_window
            try:
                visualizer_list = load_visualizer_list(season, study_class, altitudes_list, massif_names,
                                                       scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                                       year_min=year_min, year_max=year_max)

                for visualizer in visualizer_list:
                    visualizer.studies.plot_maxima_time_series(massif_names)
                massif_name = massif_names[0]
                changes_for_temporal_window = [
                    v.massif_name_to_one_fold_fit[massif_name].changes_of_moment(altitudes=[a],
                                                                                 year=year_max,
                                                                                 nb_years=nb_years,
                                                                                 order=None
                                                                                 )[0]
                    for a, v in zip(altitudes_for_return_levels, visualizer_list)]

            except WrongYearMinOrYearMax:
                changes_for_temporal_window = [np.nan for _ in altitudes_for_return_levels]

            print(changes_for_temporal_window)
            changes_in_return_levels.append(changes_for_temporal_window)
        changes_in_return_levels = np.array(changes_in_return_levels)
        all_changes_in_return_levels.append(changes_in_return_levels)
    all_changes_in_return_levels = np.array(all_changes_in_return_levels)

    return all_changes_in_return_levels, temporal_windows, altitudes_for_return_levels, nb_years


if __name__ == '__main__':
    main()
