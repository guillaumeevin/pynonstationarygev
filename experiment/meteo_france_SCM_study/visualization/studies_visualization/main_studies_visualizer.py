import time

from experiment.meteo_france_SCM_study.visualization.studies_visualization.hypercube_visualizer import \
    HypercubeVisualizer, AltitudeHypercubeVisualizer
from experiment.trend_analysis.abstract_score import MannKendall, WeigthedScore, MeanScore, MedianScore
from experiment.trend_analysis.univariate_trend_test.abstract_gev_trend_test import GevLocationTrendTest, \
    GevScaleTrendTest, GevShapeTrendTest
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import MannKendallTrendTest
from experiment.meteo_france_SCM_study.safran.safran import ExtendedSafranTotalPrecip
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies import Studies
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies_visualizer import StudiesVisualizer, \
    AltitudeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import ALL_ALTITUDES, \
    study_iterator_global, SCM_STUDIES, study_iterator

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from collections import OrderedDict


def normal_visualization():
    for study_type in [ExtendedSafranTotalPrecip]:
        extended_studies = Studies(study_type)
        studies_visualizer = StudiesVisualizer(extended_studies)
        studies_visualizer.mean_as_a_function_of_altitude(region_only=True)


def altitude_trends():
    save_to_file = True
    only_first_one = False
    # altitudes that have 20 massifs at least
    altitudes = ALL_ALTITUDES[3:-6]
    # altitudes = ALL_ALTITUDES[:2]
    for study_class in SCM_STUDIES[:]:
        for score_class in [MedianScore, MeanScore, MannKendall, WeigthedScore]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=True,
                                           score_class=score_class)
                           for study in
                           study_iterator_global(study_classes=[study_class], only_first_one=only_first_one,
                                                 altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeVisualizer(altitude_to_visualizer, multiprocessing=False, save_to_file=save_to_file)
            visualizer.negative_trend_percentages_evolution(reverse=True)


def altitude_trends_significant():
    save_to_file = False
    only_first_one = False
    # altitudes that have 20 massifs at least
    altitudes = ALL_ALTITUDES[3:-6]
    # altitudes = ALL_ALTITUDES[3:5]
    altitudes = ALL_ALTITUDES[2:4]
    for study_class in SCM_STUDIES[:1]:
        trend_test_classes = [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][:1]
        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False)
                       for study in study_iterator_global(study_classes=[study_class], only_first_one=only_first_one,
                                                          altitudes=altitudes)]
        altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
        visualizer = AltitudeVisualizer(altitude_to_visualizer, multiprocessing=False, save_to_file=save_to_file)
        visualizer.trend_tests_percentage_evolution_with_altitude(trend_test_classes, starting_year_to_weights=None)


def altitude_hypercube_test():
    save_to_file = False
    only_first_one = False
    fast = False
    altitudes = ALL_ALTITUDES[3:-6]
    altitudes = ALL_ALTITUDES[2:4]
    for study_class in SCM_STUDIES[:1]:
        trend_test_class = [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][0]
        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                       for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
        altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
        visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                         trend_class=trend_test_class, fast=fast)
        visualizer.visualize_trend_test()
        # print(visualizer.df_hypercube)


def main_run():
    # altitude_trends()
    # altitude_trends_significant()
    altitude_hypercube_test()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
