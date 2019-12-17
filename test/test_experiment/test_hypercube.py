import unittest
from collections import OrderedDict

import numpy as np

from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter.gev_trend_test_one_parameter import \
    GevLocationTrendTest
from extreme_fit.model.utils import set_seed_for_test


class TestHypercube(unittest.TestCase):
    DISPLAY = False

    def setUp(self) -> None:
        set_seed_for_test(42)
        altitudes = [900, 3000]

        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                       for study in study_iterator(study_class=SafranSnowfall, only_first_one=False,
                                                   altitudes=altitudes, verbose=self.DISPLAY)]
        self.altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
        self.trend_test_class = GevLocationTrendTest
        self.nb_data_reduced_for_speed = 4

    # def test_altitude_hypercube_visualizer(self):
    #     visualizer = AltitudeHypercubeVisualizer(self.altitude_to_visualizer, save_to_file=False,
    #                                              trend_test_class=self.trend_test_class,
    #                                              nb_data_reduced_for_speed=self.nb_data_reduced_for_speed,
    #                                              verbose=self.DISPLAY)
    #     self.df = visualizer.df_hypercube_trend_type

    def test_year_altitude_hypercube_visualizer(self):
        visualizer = Altitude_Hypercube_Year_Visualizer(self.altitude_to_visualizer, save_to_file=False,
                                                        trend_test_class=self.trend_test_class,
                                                        nb_data_reduced_for_speed=self.nb_data_reduced_for_speed,
                                                        verbose=self.DISPLAY)
        self.df = visualizer.df_hypercube_trend_type

    def tearDown(self) -> None:
        if self.DISPLAY:
            print(self.df)
        # Check that all the rows contain
        nb_non_nan_values_per_row = (~self.df.isnull()).sum(axis=1)
        equality = nb_non_nan_values_per_row.values == np.ones(len(self.df))
        self.assertTrue(equality.all())


if __name__ == '__main__':
    unittest.main()
