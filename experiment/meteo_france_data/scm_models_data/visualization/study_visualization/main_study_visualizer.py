import time
from typing import List

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.abstract_score import MannKendall
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSwe, ExtendedCrocusDepth, \
    ExtendedCrocusSwe, CrocusDaysWithSnowOnGround
from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, SafranRainfall, \
    SafranTemperature, SafranTotalPrecip

from collections import OrderedDict

from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization

SCM_STUDIES = [SafranSnowfall, CrocusSwe, CrocusDepth]
SCM_EXTENDED_STUDIES = [ExtendedSafranSnowfall, ExtendedCrocusSwe, ExtendedCrocusDepth]
SCM_STUDY_TO_EXTENDED_STUDY = OrderedDict(zip(SCM_STUDIES, SCM_EXTENDED_STUDIES))

ALL_ALTITUDES = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800]
ALL_ALTITUDES_WITHOUT_NAN = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800]
full_altitude_with_at_least_2_stations = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900,
                                          4200]
ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST = ALL_ALTITUDES[3:-6][:]

ALL_STUDIES = SCM_STUDIES + [SafranTemperature, SafranRainfall]


def study_iterator_global(study_classes, only_first_one=False, verbose=True, altitudes=None, nb_days=None) -> \
List[AbstractStudy]:
    for study_class in study_classes:
        for study in study_iterator(study_class, only_first_one, verbose, altitudes, nb_days):
            yield study
        if only_first_one:
            break


def study_iterator(study_class, only_first_one=False, verbose=True, altitudes=None, nb_consecutive_days=None) -> List[AbstractStudy]:
    # Default argument
    nb_consecutive_days = 1 if nb_consecutive_days is None else nb_consecutive_days
    altis = [1800] if altitudes is None else altitudes

    if verbose:
        print('\n\n\n\n\nLoading studies....')
    for alti in altis:
        if verbose:
            print('alti: {}, nb_day: {}     '.format(alti, nb_consecutive_days), end='')

        study = study_class(altitude=alti,
                            nb_consecutive_days=nb_consecutive_days)

        if verbose:
            massifs = study.altitude_to_massif_names[alti]
            print('{} massifs: {} \n'.format(len(massifs), massifs))
        yield study

        # Stop iterations on purpose
        if only_first_one:
            break


def extended_visualization():
    save_to_file = False
    only_first_one = True
    for study_class in SCM_EXTENDED_STUDIES[-1:]:
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file, only_one_graph=True,
                                               plot_block_maxima_quantiles=False)
            # study_visualizer.visualize_all_mean_and_max_graphs()
            study_visualizer.visualize_all_experimental_law()
    # for study_class in SCM_EXTENDED_STUDIES[:]:
    #     for study in study_iterator(study_class, only_first_one=False):
    #         study_visualizer = StudyVisualizer(study, single_massif_graph=True, save_to_file=True)
    #         # study_visualizer.visualize_all_kde_graphs()
    #         study_visualizer.visualize_all_experimental_law()


def annual_mean_vizu_compare_durand_study(safran=True, take_mean_value=True, altitude=1800):
    if safran:
        for study_class in [SafranTotalPrecip, SafranRainfall, SafranSnowfall, SafranTemperature][-1:]:
            study = study_class(altitude=altitude, year_min=1958, year_max=2002)
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_annual_mean_values(take_mean_value=True)
    else:
        for study_class in [CrocusSwe, CrocusDepth, CrocusDaysWithSnowOnGround][-1:]:
            study = study_class(altitude=altitude, year_min=1958, year_max=2005)
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_annual_mean_values(take_mean_value=take_mean_value)


def max_stable_process_vizu_compare_gaume_study(altitude=1800, nb_days=1):
    study = SafranSnowfall(altitude=altitude, nb_consecutive_days=nb_days)
    study_visualizer = StudyVisualizer(study)
    study_visualizer.visualize_brown_resnick_fit()


def normal_visualization(temporal_non_stationarity=False):
    save_to_file = False
    only_first_one = True
    # for study_class in SCM_STUDIES[:1]:
    for study_class in [CrocusDepth, SafranSnowfall, SafranRainfall, SafranTemperature][1:2]:
        for study in study_iterator(study_class, only_first_one=only_first_one, altitudes=[300]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               temporal_non_stationarity=temporal_non_stationarity)
            print(study_visualizer.massif_name_to_scores)
            # study_visualizer.visualize_all_mean_and_max_graphs()

            # study_visualizer.visualize_independent_margin_fits(threshold=[None, 20, 40, 60][0])
            # study_visualizer.visualize_annual_mean_values()


def all_normal_vizu():
    for study in study_iterator_global(study_classes=ALL_STUDIES, only_first_one=False, altitudes=ALL_ALTITUDES):
        study_visualizer = StudyVisualizer(study, save_to_file=True, temporal_non_stationarity=True)
        study_visualizer.visualize_all_mean_and_max_graphs()

def scores_vizu():
    save_to_file = False
    only_first_one = True
    for study in study_iterator_global(study_classes=ALL_STUDIES, only_first_one=only_first_one, altitudes=[1800]):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file, temporal_non_stationarity=True)
        # study_visualizer.visualize_all_score_wrt_starting_year()
        study_visualizer.visualize_all_score_wrt_starting_year()


def all_scores_vizu():
    save_to_file = True
    only_first_one = False
    for study in study_iterator_global(study_classes=[SafranSnowfall], only_first_one=only_first_one, altitudes=ALL_ALTITUDES):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file, temporal_non_stationarity=True, verbose=True)
        # study_visualizer.visualize_all_mean_and_max_graphs()
        study_visualizer.visualize_all_score_wrt_starting_year()

def complete_analysis(only_first_one=False):
    """An overview of everything that is possible with study OR extended study"""
    for study_class, extended_study_class in list(SCM_STUDY_TO_EXTENDED_STUDY.items())[:]:
        # First explore everything you can do with the extended study class
        print('Extended study')
        for extended_study in study_iterator(extended_study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(extended_study, save_to_file=True)
            study_visualizer.visualize_all_mean_and_max_graphs()
            study_visualizer.visualize_all_experimental_law()
        print('Study normal')
        for study in study_iterator(study_class, only_first_one=only_first_one):
            study_visualizer = StudyVisualizer(study, save_to_file=True)
            study_visualizer.visualize_linear_margin_fit()


def trend_analysis():
    save_to_file = True
    only_first_one = False
    short_altitudes = [300, 1200, 2100, 3000][:1]
    full_altitude_with_at_least_2_stations = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
                                              3900, 4200][:]
    durand_altitude = [1800]
    altitudes = durand_altitude
    normalization_class = [None, BetweenMinusOneAndOneNormalization, BetweenZeroAndOneNormalization][-1]
    study_classes = [CrocusSwe, CrocusDepth, SafranSnowfall, SafranRainfall, SafranTemperature][2:3]
    for study in study_iterator_global(study_classes, only_first_one=only_first_one, altitudes=altitudes):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                           transformation_class=normalization_class,
                                           temporal_non_stationarity=True,
                                           verbose=True,
                                           multiprocessing=True,
                                           complete_non_stationary_trend_analysis=True)
        # study_visualizer.visualize_all_independent_temporal_trend()
        # study_visualizer.visualize_temporal_trend_relevance()
        study_visualizer.df_trend_spatio_temporal(GevLocationChangePointTest,
                                                  starting_years=[1958, 1980], nb_massif_for_fast_mode=2)

def maxima_analysis():
    save_to_file = False
    only_first_one = True
    durand_altitude = [1800]
    altitudes = durand_altitude
    normalization_class = BetweenZeroAndOneNormalization
    study_classes = [ SafranSnowfall][:]
    for study in study_iterator_global(study_classes, only_first_one=only_first_one, altitudes=altitudes):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                           transformation_class=normalization_class,
                                           temporal_non_stationarity=True,
                                           verbose=True,
                                           multiprocessing=True,
                                           complete_non_stationary_trend_analysis=True,
                                           score_class=MannKendall)
        study_visualizer.visualize_all_score_wrt_starting_year()
        # study_visualizer.visualize_all_independent_temporal_trend()
        # study_visualizer.visualize_all_mean_and_max_graphs()

def main_run():
    # normal_visualization(temporal_non_stationarity=True)
    trend_analysis()
    # all_scores_vizu()
    # maxima_analysis()
    # all_normal_vizu()

    # annual_mean_vizu_compare_durand_study(safran=True, take_mean_value=True, altitude=2100)

    # max_stable_process_vizu_compare_gaume_study(altitude=1800, nb_days=1)
    # extended_visualization()
    # complete_analysis()
    # scores_vizu()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))