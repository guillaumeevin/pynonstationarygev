import unittest
from collections import OrderedDict

from experiment.eurocode_data.eurocode_return_level_uncertainties import EurocodeLevelUncertaintyFromExtremes
from experiment.eurocode_data.eurocode_visualizer import plot_model_name_to_dep_to_ordered_return_level_uncertainties
from experiment.eurocode_data.massif_name_to_departement import DEPARTEMENT_TYPES
from experiment.eurocode_data.utils import EUROCODE_ALTITUDES


class TestCoordinateSensitivity(unittest.TestCase):
    DISPLAY = False

    def test_departement_eurocode_plot(self):
        # Create an example
        example = EurocodeLevelUncertaintyFromExtremes(posterior_mean=1.0,
                                                       poster_uncertainty_interval=(0.5, 1.25))
        dep_to_model_name_toreturn_level_uncertainty = {
            dep: {"example": [example for _ in EUROCODE_ALTITUDES]} for dep in DEPARTEMENT_TYPES
        }
        plot_model_name_to_dep_to_ordered_return_level_uncertainties(dep_to_model_name_toreturn_level_uncertainty,
                                                                     show=self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
