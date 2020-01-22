from enum import Enum

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from root_utils import get_display_name_from_object_type

paper_altitudes = ALL_ALTITUDES_WITHOUT_NAN
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode, CrocusSnowLoad3Days][:2]
# dpi_paper1_figure = 700
dpi_paper1_figure = None

class ModelSubsetForUncertainty(Enum):
    stationary_gumbel = 0
    stationary_gumbel_and_gev = 1
    non_stationary_gumbel = 2
    non_stationary_gumbel_and_gev = 3



