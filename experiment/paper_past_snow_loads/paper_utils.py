from enum import Enum

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSnowLoadEurocode, \
    CrocusSnowLoad3Days
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from root_utils import get_display_name_from_object_type

# paper_altitudes = [300, 600, 900, 1200, 1500, 1800, 2100, 2400, 2700]
paper_altitudes = ALL_ALTITUDES_WITHOUT_NAN
paper_study_classes = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode, CrocusSnowLoad3Days]
# dpi_paper1_figure = 700
dpi_paper1_figure = None

class ModelSubsetForUncertainty(Enum):
    stationary_gumbel = 0
    stationary_gumbel_and_gev = 1
    non_stationary_gumbel = 2
    non_stationary_gumbel_and_gev = 3



