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
        example1 = EurocodeLevelUncertaintyFromExtremes(posterior_mean=1.0,
                                                       poster_uncertainty_interval=(0.5, 1.25))
        example2 = EurocodeLevelUncertaintyFromExtremes(posterior_mean=0.2,
                                                       poster_uncertainty_interval=(0.1, 0.35))
        example3 = EurocodeLevelUncertaintyFromExtremes(posterior_mean=0.4,
                                                       poster_uncertainty_interval=(0.25, 0.6))
        altitude_examples = EUROCODE_ALTITUDES[:2]
        dep_to_model_name_toreturn_level_uncertainty = {
            dep: {"example1": [example1 for _ in altitude_examples],
                  "example2": [example2 for _ in altitude_examples],
                  "example3": [example3 for _ in altitude_examples],
                  } for dep in DEPARTEMENT_TYPES
        }
        plot_model_name_to_dep_to_ordered_return_level_uncertainties(dep_to_model_name_toreturn_level_uncertainty,
                                                                     altitudes=altitude_examples,
                                                                     show=self.DISPLAY)


if __name__ == '__main__':
    unittest.main()
