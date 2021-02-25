import unittest


class TestResults(unittest.TestCase):
    pass

    # def test_run_intermediate_results(self):
    #     plt.close()
    #     # Load data
    #     altitude_to_visualizer = load_altitude_to_visualizer(altitudes=[300, 600], massif_names=None,
    #                                                          study_class=CrocusSnowLoadTotal,
    #                                                          model_subsets_for_uncertainty=None,
    #                                                          uncertainty_methods=[
    #                                                              ConfidenceIntervalMethodFromExtremes.ci_mle])
    #     plot_trend_map(altitude_to_visualizer)
    #     plot_trend_curves(altitude_to_visualizer)
    #     plot_uncertainty_massifs(altitude_to_visualizer)
    #     plot_uncertainty_histogram(altitude_to_visualizer)
    #     plot_selection_curves(altitude_to_visualizer)
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
