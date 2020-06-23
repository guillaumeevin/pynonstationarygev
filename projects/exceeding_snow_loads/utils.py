from enum import Enum

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from extreme_trend.trend_test_one_parameter.gev_trend_test_one_parameter import GevVersusGev, GevScaleTrendTest, \
    GevLocationTrendTest, GevShapeTrendTest
from extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter import \
    GumbelVersusGumbel, GumbelLocationTrendTest, GumbelScaleTrendTest, GevStationaryVersusGumbel
from extreme_trend.trend_test_three_parameters.gev_trend_test_three_parameters import \
    GevLocationAndScaleTrendTestAgainstGumbel, GevLocationAndScaleAndShapeTrendTest, \
    GevLocationQuadraticTrendTestAgainstGumbel, \
    GevScaleQuadraticTrendTestAgainstGumbel
from extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters import \
    GevLocationAgainstGumbel, GevScaleAgainstGumbel, GevLocationAndScaleTrendTest, GevScaleAndShapeTrendTest, \
    GevLocationAndShapeTrendTest, GevQuadraticLocationTrendTest, GevQuadraticScaleTrendTest
from extreme_trend.trend_test_two_parameters.gumbel_test_two_parameters import \
    GumbelLocationAndScaleTrendTest, GumbelLocationQuadraticTrendTest, GumbelScaleQuadraticTrendTest

paper_altitudes = ALL_ALTITUDES_WITHOUT_NAN
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode, CrocusSnowLoad3Days][:2]
# dpi_paper1_figure = 700
dpi_paper1_figure = None
NON_STATIONARY_TREND_TEST_PAPER_1 = [GumbelVersusGumbel,
                                     GumbelLocationTrendTest, GumbelScaleTrendTest,
                                     GumbelLocationAndScaleTrendTest,
                                     GevStationaryVersusGumbel,
                                     GevLocationAgainstGumbel, GevScaleAgainstGumbel,
                                     GevLocationAndScaleTrendTestAgainstGumbel]

NON_STATIONARY_TREND_TEST_PAPER_2 = [
    # Gumbel models
    #GumbelVersusGumbel,
    #GumbelLocationTrendTest,
    #GumbelScaleTrendTest,
    #GumbelLocationAndScaleTrendTest,
    # GEV models with constant shape
    #GevVersusGev,
    GevLocationTrendTest,
    # GevScaleTrendTest,
    #GevLocationAndScaleTrendTest,
    # GEV models with linear shape
    #GevShapeTrendTest,
    #GevLocationAndShapeTrendTest, GevScaleAndShapeTrendTest, GevLocationAndScaleAndShapeTrendTest,
    # Quadratic model for the Gev/Gumbel and for the location/scale
    #GevQuadraticLocationTrendTest, GevQuadraticScaleTrendTest, GumbelLocationQuadraticTrendTest, GumbelScaleQuadraticTrendTest,
]


def get_trend_test_name(trend_test_class):
    years = list(range(10))
    trend_test = trend_test_class(years, years, None)
    return trend_test.name


if __name__ == '__main__':
    for trend_test_class in NON_STATIONARY_TREND_TEST_PAPER_2:
        print(get_trend_test_name(trend_test_class))
