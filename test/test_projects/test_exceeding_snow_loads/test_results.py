import unittest

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_trend.visualizers.utils import load_altitude_to_visualizer
from projects.exceeding_snow_loads.section_results.plot_selection_curves import plot_selection_curves
from projects.exceeding_snow_loads.section_results.plot_trend_curves import plot_trend_curves, plot_trend_map
from projects.exceeding_snow_loads.section_results.plot_uncertainty_curves import plot_uncertainty_massifs
from projects.exceeding_snow_loads.section_results.plot_uncertainty_histogram import plot_uncertainty_histogram


class TestResults(unittest.TestCase):

    def test_run_intermediate_results(self):
        # Load data
        altitude_to_visualizer = load_altitude_to_visualizer(altitudes=[300, 600], massif_names=None,
                                                             study_class=CrocusSnowLoadTotal,
                                                             model_subsets_for_uncertainty=None,
                                                             uncertainty_methods=[
                                                                 ConfidenceIntervalMethodFromExtremes.ci_mle])
        plot_trend_map(altitude_to_visualizer)
        plot_trend_curves(altitude_to_visualizer)
        plot_uncertainty_massifs(altitude_to_visualizer)
        plot_uncertainty_histogram(altitude_to_visualizer)
        plot_selection_curves(altitude_to_visualizer)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
