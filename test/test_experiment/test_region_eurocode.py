import unittest
from collections import OrderedDict

from experiment.eurocode_data.eurocode_visualizer import display_region_limit
from experiment.eurocode_data.region_eurocode import C1, E
from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSweTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.non_stationary_trends import \
    ConditionalIndedendenceLocationTrendTest
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenZeroAndOneNormalizationMinEpsilon, BetweenZeroAndOneNormalizationMaxEpsilon
from utils import get_display_name_from_object_type


class TestCoordinateSensitivity(unittest.TestCase):
    DISPLAY = False

    def test_region_eurocode(self):
        altitudes = [900, 1200, 1500, 1800]
        ordered_massif_name_to_quantiles = OrderedDict()
        ordered_massif_name_to_quantiles['Vanoise'] = [1.2, 1.5, 1.7, 2.1]
        ordered_massif_name_to_quantiles['Vercors'] = [0.7, 0.8, 1.1, 1.5]
        display_region_limit(C1, altitudes, ordered_massif_name_to_quantiles, display=self.DISPLAY)
        display_region_limit(E, altitudes, ordered_massif_name_to_quantiles, display=self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
