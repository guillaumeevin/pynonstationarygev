import time
from collections import OrderedDict
from typing import List

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall, AdamontPrecipitation
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth, CrocusSweTotal, \
    ExtendedCrocusDepth, \
    ExtendedCrocusSweTotal, CrocusDaysWithSnowOnGround, CrocusSwe3Days, CrocusSnowLoad3Days, CrocusSnowLoadTotal, \
    CrocusSnowLoadEurocode, CrocusSnowLoad5Days, CrocusSnowLoad7Days
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2020, CrocusSnowLoad2019
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_snow_density import CrocusSnowDensity
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDensityVariable
from extreme_data.meteo_france_data.scm_models_data.safran.gap_between_study import GapBetweenSafranSnowfall2019And2020, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered, GapBetweenSafranSnowfall2019AndMySafranSnowfall2019, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered, \
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019RecenteredMeanRate
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall, ExtendedSafranSnowfall, \
    SafranRainfall, \
    SafranTemperature, SafranPrecipitation, SafranSnowfall1Day, SafranSnowfall3Days, SafranSnowfall5Days, \
    SafranSnowfall7Days, SafranPrecipitation1Day, SafranPrecipitation3Days, SafranPrecipitation5Days, \
    SafranPrecipitation7Days, SafranDateFirstSnowfall, SafranSnowfallCenterOnDay1dayMeanRate, \
    SafranSnowfallCenterOnDay1day
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_precipf import SafranPrecipitation2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020, \
    SafranSnowfall2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariableCenterOnDay
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
from root_utils import get_display_name_from_object_type
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization

snow_density_str = '$\\rho_{SNOW}$'
eurocode_snow_density = '{}=150 {}'.format(snow_density_str, CrocusDensityVariable.UNIT)
SLEurocode = 'SL from max HS with ' + eurocode_snow_density

SCM_STUDIES = [SafranSnowfall, CrocusSweTotal, CrocusDepth, CrocusSwe3Days]
SCM_STUDIES_NAMES = [get_display_name_from_object_type(k) for k in SCM_STUDIES]
SCM_STUDY_NAME_TO_SCM_STUDY = dict(zip(SCM_STUDIES_NAMES, SCM_STUDIES))

SCM_STUDY_CLASS_TO_ABBREVIATION = {
    SafranSnowfall: 'SF3',
    SafranSnowfall1Day: 'daily snowfall',
    SafranSnowfall2020: 'daily snowfall',
    SafranSnowfall2019: 'daily snowfall',
    SafranSnowfallCenterOnDay1dayMeanRate: 'daily snowfall',
    SafranSnowfallCenterOnDay1day: 'daily snowfall',
    GapBetweenSafranSnowfall2019And2020: 'daily snowfall\n bias = SAFRAN 2020 minus SAFRAN 2019',
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered: 'daily snowfall\n my SAFRAN 2019 recentered minus SAFRAN 2019',
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019NotRecentered: 'daily snowfall\n my SAFRAN 2019 notrecentered minus SAFRAN 2019',
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019: 'daily snowfall\n my SAFRAN 2019 minus SAFRAN 2019',
    GapBetweenSafranSnowfall2019AndMySafranSnowfall2019RecenteredMeanRate: 'daily snowfall\n my SAFRAN 2019 recentered mean rate minus SAFRAN 2019',
    SafranSnowfall3Days: 'SF3',
    SafranSnowfall5Days: 'SF5',
    SafranSnowfall7Days: 'SF7',
    SafranPrecipitation1Day: 'Precipitation',
    SafranTemperature: 'Temperature',
    SafranPrecipitation2019: 'precipitation',
    SafranPrecipitation3Days: 'PR3',
    SafranPrecipitation5Days: 'PR5',
    SafranPrecipitation7Days: 'PR7',
    CrocusSweTotal: 'SWE',
    CrocusSwe3Days: 'SWE3',
    CrocusSnowLoadEurocode: 'GSL from annual maximum of HS \nand {}'.format(eurocode_snow_density),
    CrocusDepth: 'SD',
    CrocusSnowLoadTotal: 'GSL',
    CrocusSnowLoad2019: 'GSL',
    CrocusSnowLoad2020: 'GSL',
    CrocusSnowLoad3Days: 'GSL3',
    CrocusSnowLoad5Days: 'GSL5',
    CrocusSnowLoad7Days: 'GSL7',
    CrocusSnowDensity: 'Density',
    SafranDateFirstSnowfall: 'SF1 first date'
}
# I keep the scm study separated from the adamont study (for the tests)
ADAMONT_STUDY_CLASS_TO_ABBREVIATION = {
    AdamontSnowfall: 'daily snowfall',
    AdamontSnowLoad: 'snow load',
    AdamontPrecipitation: 'precipitation',
}
STUDY_CLASS_TO_ABBREVIATION = {**ADAMONT_STUDY_CLASS_TO_ABBREVIATION, **SCM_STUDY_CLASS_TO_ABBREVIATION}

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


















