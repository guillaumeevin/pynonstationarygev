import time
from collections import OrderedDict

from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import ALL_ALTITUDES, \
    SCM_STUDIES, study_iterator
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_trend_test.abstract_gev_trend_test import GevLocationTrendTest, \
    GevScaleTrendTest, GevShapeTrendTest
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import MannKendallTrendTest


def altitude_trend_with_hypercube():
    save_to_file = True
    only_first_one = False
    fast = False
    altitudes = ALL_ALTITUDES[3:-6]
    # altitudes = ALL_ALTITUDES[2:4]
    for study_class in SCM_STUDIES[:]:
        for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][:]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                           for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                     trend_test_class=trend_test_class, fast=fast)
            visualizer.visualize_altitude_trend_test()


def spatial_trend_with_hypercube():
    save_to_file = False
    only_first_one = False
    fast = True
    # altitudes = ALL_ALTITUDES[3:-6]
    altitudes = ALL_ALTITUDES[2:4]
    for study_class in SCM_STUDIES[:1]:
        for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][:1]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                           for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                     trend_test_class=trend_test_class, fast=fast)
            visualizer.visualize_massif_trend_test()
            visualizer.visualize_year_trend_test()
            visualizer.visualize_altitude_trend_test()



def main_run():
    # altitude_trends()
    # altitude_trends_significant()
    spatial_trend_with_hypercube()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
