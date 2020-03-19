from enum import Enum

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from extreme_trend.trend_test_one_parameter.gumbel_trend_test_one_parameter import \
    GumbelVersusGumbel, GumbelLocationTrendTest, GumbelScaleTrendTest, GevStationaryVersusGumbel
from extreme_trend.trend_test_three_parameters.gev_trend_test_three_parameters import \
    GevLocationAndScaleTrendTestAgainstGumbel
from extreme_trend.trend_test_two_parameters.gev_trend_test_two_parameters import \
    GevLocationAgainstGumbel, GevScaleAgainstGumbel
from extreme_trend.trend_test_two_parameters.gumbel_test_two_parameters import \
    GumbelLocationAndScaleTrendTest

paper_altitudes = ALL_ALTITUDES_WITHOUT_NAN
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode, CrocusSnowLoad3Days][:2]
# dpi_paper1_figure = 700
dpi_paper1_figure = None
NON_STATIONARY_TREND_TEST_PAPER = [GumbelVersusGumbel,
                                   GumbelLocationTrendTest, GumbelScaleTrendTest,
                                   GumbelLocationAndScaleTrendTest,
                                   GevStationaryVersusGumbel,
                                   GevLocationAgainstGumbel, GevScaleAgainstGumbel,
                                   GevLocationAndScaleTrendTestAgainstGumbel]



