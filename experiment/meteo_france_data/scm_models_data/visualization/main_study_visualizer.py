import time
from typing import List

from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDensityVariable
from experiment.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
from projects.exceeding_snow_loads.discussion_data_comparison_with_eurocode.crocus_study_comparison_with_eurocode import \
    CrocusDifferenceSnowLoad, \
    CrocusSnowDensityAtMaxofSwe, CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization, \
    CrocusSnowDepthDifference, CrocusSnowDepthAtMaxofSwe
from experiment.trend_analysis.abstract_score import MannKendall
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSweTotal, ExtendedCrocusDepth, \
    ExtendedCrocusSweTotal, CrocusDaysWithSnowOnGround, CrocusSwe3Days, CrocusSnowLoad3Days, CrocusSnowLoadTotal, \
    CrocusSnowLoadEurocode, CrocusSnowLoad5Days, CrocusSnowLoad7Days
from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, \
    SafranRainfall, \
    SafranTemperature, SafranPrecipitation

from collections import OrderedDict

from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevLocationTrendTest
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization
from root_utils import get_display_name_from_object_type

snow_density_str = '$\\rho_{SNOW}$'
eurocode_snow_density = '{}=150 {}'.format(snow_density_str, CrocusDensityVariable.UNIT)
SLEurocode = 'SL from max HS with ' + eurocode_snow_density

SCM_STUDIES = [SafranSnowfall, CrocusSweTotal, CrocusDepth, CrocusSwe3Days]
SCM_STUDIES_NAMES = [get_display_name_from_object_type(k) for k in SCM_STUDIES]
SCM_STUDY_NAME_TO_SCM_STUDY = dict(zip(SCM_STUDIES_NAMES, SCM_STUDIES))
SCM_STUDY_CLASS_TO_ABBREVIATION = {
    SafranSnowfall: 'SF3',
    CrocusSweTotal: 'SWE',
    CrocusSwe3Days: 'SWE3',
    CrocusSnowLoadEurocode: 'GSL from annual maximum of HS \nand {}'.format(eurocode_snow_density),
    CrocusDepth: 'SD',
    CrocusSnowLoadTotal: 'GSL',
    CrocusSnowLoad3Days: 'GSL3',
    CrocusSnowLoad5Days: 'GSL5',
    CrocusSnowLoad7Days: 'GSL7',
    CrocusSnowDensityAtMaxofSwe: '{} when the max of GSL \nis reached'.format(snow_density_str),
    CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization: 'max GSL rescaled - GSL from max HS \nboth with {}'.format(eurocode_snow_density),
    CrocusDifferenceSnowLoad: ('max GSL - GSL from max HS \n with {}'.format(eurocode_snow_density)),
    CrocusSnowDepthDifference: 'max HS - HS at max of GSL',
    CrocusSnowDepthAtMaxofSwe: 'HS at max of GSL',
}

altitude_massif_name_and_study_class_for_poster = [
    (900, 'Chartreuse', CrocusSweTotal),
    (1800, 'Vanoise', CrocusDepth),
    (2700, 'Parpaillon', SafranSnowfall),
]

altitude_massif_name_and_study_class_for_poster_evan = [
    (900, 'Chartreuse', CrocusSwe3Days),
    (1800, 'Vanoise', CrocusSwe3Days),
    (2700, 'Parpaillon', CrocusSwe3Days),
]

altitude_massif_name_and_study_class_for_committee = [
    (900, 'Chartreuse', CrocusSnowLoad3Days),
    (1800, 'Vanoise', CrocusSnowLoad3Days),
    (2700, 'Parpaillon', CrocusSnowLoad3Days),
]

SCM_STUDY_NAME_TO_ABBREVIATION = {get_display_name_from_object_type(k): v for k, v in
                                  SCM_STUDY_CLASS_TO_ABBREVIATION.items()}
SCM_COLORS = ['tab:orange', 'y', 'tab:purple', 'lightseagreen']
SCM_STUDY_CLASS_TO_COLOR = dict(zip(SCM_STUDIES, SCM_COLORS))
SCM_STUDY_NAME_TO_COLOR = {get_display_name_from_object_type(s): color
                           for s, color in zip(SCM_STUDIES, SCM_COLORS)}
poster_altitude_to_color = dict(zip([900, 1800, 2700], ['y', 'tab:purple', 'tab:orange']))
SCM_EXTENDED_STUDIES = [ExtendedSafranSnowfall, ExtendedCrocusSweTotal, ExtendedCrocusDepth]
SCM_STUDY_TO_EXTENDED_STUDY = OrderedDict(zip(SCM_STUDIES, SCM_EXTENDED_STUDIES))

ALL_ALTITUDES = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500, 4800]
ALTITUDES_LOW_MIDDLE_HIGH = [900, 1800, 2700]
ALL_ALTITUDES_WITHOUT_NAN = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900, 4200, 4500,
                             4800]
full_altitude_with_at_least_2_stations = [0, 300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600, 3900,
                                          4200]
ALL_ALTITUDES_WITH_20_STATIONS_AT_LEAST = ALL_ALTITUDES[3:-6][:]

ALL_STUDIES = SCM_STUDIES + [SafranTemperature, SafranRainfall]


def study_iterator_global(study_classes, only_first_one=False, verbose=True, altitudes=None, nb_days=None,
                          orientations=None) -> \
        List[AbstractStudy]:
    for study_class in study_classes:
        for study in study_iterator(study_class, only_first_one, verbose, altitudes, nb_days,
                                    orientations=orientations):
            yield study
        if only_first_one:
            break


def study_iterator(study_class, only_first_one=False, verbose=True, altitudes=None, nb_consecutive_days=3,
                   orientations=None) -> List[
    AbstractStudy]:
    # Default argument
    altis = [1800] if altitudes is None else altitudes
    orients = [None] if orientations is None else orientations

    if verbose:
        print('\n\n\n\n\nLoading studies....')
    for alti in altis:
        for orient in orients:
            if verbose:
                print('alti: {}, nb_day: {}  orient = {}   '.format(alti, nb_consecutive_days, orient), end='')

            study = study_class(altitude=alti, orientation=orient)

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
        for study_class in [SafranPrecipitation, SafranRainfall, SafranSnowfall, SafranTemperature][-1:]:
            study = study_class(altitude=altitude, year_min=1958, year_max=2002)
            study_visualizer = StudyVisualizer(study)
            study_visualizer.visualize_annual_mean_values(take_mean_value=True)
    else:
        for study_class in [CrocusSweTotal, CrocusDepth, CrocusDaysWithSnowOnGround][-1:]:
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


def case_study():
    for study in study_iterator(study_class=SafranSnowfall, only_first_one=False, altitudes=[2100],
                                nb_consecutive_days=3):
        study_visualizer = StudyVisualizer(study, save_to_file=False, temporal_non_stationarity=False)
        study_visualizer.visualize_all_mean_and_max_graphs()
        massif_id = study.study_massif_names.index('Chablais')
        print(massif_id)
        x, y = study_visualizer.smooth_maxima_x_y(massif_id)
        print(x)
        print(y)


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
    for study in study_iterator_global(study_classes=[SafranSnowfall], only_first_one=only_first_one,
                                       altitudes=ALL_ALTITUDES):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file, temporal_non_stationarity=True,
                                           verbose=True)
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
    study_classes = [CrocusSweTotal, CrocusDepth, SafranSnowfall, SafranRainfall, SafranTemperature][2:3]
    for study in study_iterator_global(study_classes, only_first_one=only_first_one, altitudes=altitudes):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                           transformation_class=normalization_class,
                                           temporal_non_stationarity=True,
                                           verbose=True,
                                           multiprocessing=True,
                                           complete_non_stationary_trend_analysis=True)
        # study_visualizer.visualize_all_independent_temporal_trend()
        # study_visualizer.visualize_temporal_trend_relevance()
        study_visualizer.df_trend_spatio_temporal(GevLocationTrendTest,
                                                  starting_years=[1958, 1980], nb_massif_for_fast_mode=2)


def maxima_analysis():
    save_to_file = False
    only_first_one = True
    durand_altitude = [2700]
    altitudes = durand_altitude
    normalization_class = BetweenZeroAndOneNormalization
    study_classes = [CrocusSwe3Days][:]
    for study in study_iterator_global(study_classes, only_first_one=only_first_one, altitudes=altitudes):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                           transformation_class=normalization_class,
                                           temporal_non_stationarity=True,
                                           verbose=True,
                                           multiprocessing=True,
                                           complete_non_stationary_trend_analysis=True,
                                           score_class=MannKendall)
        # study_visualizer.visualize_all_score_wrt_starting_year()
        # study_visualizer.visualize_all_independent_temporal_trend()
        # study_visualizer.visualize_independent_margin_fits()
        # study_visualizer.visualize_all_mean_and_max_graphs()
        study_visualizer.visualize_summary_of_annual_values_and_stationary_gev_fit()


def max_graph_annual_maxima_poster():
    save_to_file = True
    choice_tuple = [
        altitude_massif_name_and_study_class_for_poster,
        altitude_massif_name_and_study_class_for_poster_evan,
        altitude_massif_name_and_study_class_for_committee,
    ][2]
    for altitude, massif_name, study_class in choice_tuple:
        for study in study_iterator_global([study_class], altitudes=[altitude]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
            # color = SCM_STUDY_CLASS_TO_COLOR[study_class]
            color = poster_altitude_to_color[altitude]
            study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color)
            # study_visualizer.visualize_gev_graphs_poster(massif_name, altitude, snow_abbreviation, color)


def altitude_analysis():
    study = CrocusSweTotal(altitude=900)
    all_names = set(study.study_massif_names)
    for a, names in study.altitude_to_massif_names.items():
        print(a, len(names), all_names - set(names))


def main_run():
    # normal_visualization(temporal_non_stationarity=True)
    # trend_analysis()

    # altitude_analysis()
    max_graph_annual_maxima_poster()
    # maxima_analysis()
    # case_study()
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
