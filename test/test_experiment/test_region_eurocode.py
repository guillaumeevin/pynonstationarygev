import unittest
from collections import OrderedDict

from experiment.eurocode_data.eurocode_visualizer import display_region_limit
from experiment.eurocode_data.eurocode_region import C1, E, C2


class TestCoordinateSensitivity(unittest.TestCase):
    DISPLAY = False

    def test_region_eurocode(self):
        altitudes = [900, 1200, 1500, 1800]
        ordered_massif_name_to_quantiles = OrderedDict()
        ordered_massif_name_to_quantiles['Vanoise'] = [1.2, 1.5, 1.7, 2.1]
        ordered_massif_name_to_quantiles['Vercors'] = [0.7, 0.8, 1.1, 1.5]
        display_region_limit(C1, altitudes, ordered_massif_name_to_quantiles, display=self.DISPLAY)
        display_region_limit(C2, altitudes, ordered_massif_name_to_quantiles, display=self.DISPLAY)
        display_region_limit(E, altitudes, ordered_massif_name_to_quantiles, display=self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
