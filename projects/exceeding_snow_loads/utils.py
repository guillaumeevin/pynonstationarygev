from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from extreme_trend.trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import GevVersusGev, GevScaleTrendTest, \
    GevLocationTrendTest
from extreme_trend.trend_test.trend_test_one_parameter.gumbel_trend_test_one_parameter import GumbelVersusGumbel, \
    GevStationaryVersusGumbel, GumbelLocationTrendTest, GumbelScaleTrendTest
from extreme_trend.trend_test.trend_test_three_parameters.gev_trend_test_three_parameters import \
    GevLocationAndScaleTrendTestAgainstGumbel
from extreme_trend.trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import GevLocationAgainstGumbel, \
    GevScaleAgainstGumbel, GevLocationAndScaleTrendTest, GevQuadraticLocationTrendTest, GevQuadraticScaleTrendTest
from extreme_trend.trend_test.trend_test_two_parameters.gumbel_test_two_parameters import \
    GumbelLocationAndScaleTrendTest, GumbelLocationQuadraticTrendTest, GumbelScaleQuadraticTrendTest

paper_altitudes = ALL_ALTITUDES_WITHOUT_NAN
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode][:]
# dpi_paper1_figure = 700
dpi_paper1_figure = None
NON_STATIONARY_TREND_TEST_PAPER_1 = [GumbelVersusGumbel,
                                     GevStationaryVersusGumbel,

                                     GumbelLocationTrendTest,
                                     GevLocationAgainstGumbel,

                                     GumbelScaleTrendTest,
                                     GevScaleAgainstGumbel,

                                     GumbelLocationAndScaleTrendTest,
                                     GevLocationAndScaleTrendTestAgainstGumbel,

                                     ]

ALTITUDE_TO_GREY_MASSIF = {
    300: ['Devoluy', 'Queyras', 'Champsaur', 'Grandes-Rousses', 'Ubaye', 'Pelvoux', 'Haute-Maurienne', 'Maurienne', 'Vanoise', 'Thabor', 'Mercantour', 'Haut_Var-Haut_Verdon', 'Haute-Tarentaise', 'Parpaillon', 'Belledonne', 'Oisans'],
    600: ['Queyras', 'Pelvoux', 'Haute-Maurienne', 'Mercantour', 'Haut_Var-Haut_Verdon', 'Thabor'],
    900: [],
    1200: [],
    1500: [],
    1800: [],
}


def get_trend_test_name(trend_test_class):
    years = list(range(10))
    trend_test = trend_test_class(years, years, None)
    return trend_test.name
