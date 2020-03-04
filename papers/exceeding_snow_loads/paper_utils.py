from enum import Enum

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter.gumbel_trend_test_one_parameter import \
    GumbelVersusGumbel, GumbelLocationTrendTest, GumbelScaleTrendTest, GevStationaryVersusGumbel
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_three_parameters.gev_trend_test_three_parameters import \
    GevLocationAndScaleTrendTestAgainstGumbel
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_two_parameters.gev_trend_test_two_parameters import \
    GevLocationAgainstGumbel, GevScaleAgainstGumbel
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_two_parameters.gumbel_test_two_parameters import \
    GumbelLocationAndScaleTrendTest
from root_utils import get_display_name_from_object_type

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



class ModelSubsetForUncertainty(Enum):
    stationary_gumbel = 0
    stationary_gumbel_and_gev = 1
    non_stationary_gumbel = 2
    non_stationary_gumbel_and_gev = 3
    stationary_gev = 4
