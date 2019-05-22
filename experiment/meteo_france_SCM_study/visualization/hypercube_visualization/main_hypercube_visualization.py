import time
from itertools import product
from collections import OrderedDict

from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.main_study_visualizer import ALL_ALTITUDES, \
    SCM_STUDIES, study_iterator, study_iterator_global
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_trend_test.abstract_gev_trend_test import GevLocationTrendTest, \
    GevScaleTrendTest, GevShapeTrendTest
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import MannKendallTrendTest
from utils import get_display_name_from_object_type


# def full_trends_with_altitude_hypercube():
#     save_to_file = True
#     only_first_one = False
#     fast = False
#     altitudes = ALL_ALTITUDES[3:-6]
#     for study_class in SCM_STUDIES[:]:
#         for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][:]:
#             visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
#                            for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
#                                                        altitudes=altitudes)]
#             altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
#             visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
#                                                      trend_test_class=trend_test_class, fast=fast)
#             visualizer.visualize_massif_trend_test()
#             visualizer.visualize_year_trend_test()
#             visualizer.visualize_altitude_trend_test()


def full_trends_with_quantity_altitude_hypercube():
    save_to_file = True
    only_first_one = False
    fast = False
    add_detailed_plots = False
    altitudes = ALL_ALTITUDES[3:-6]
    study_classes = SCM_STUDIES
    for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][1:2]:
        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                       for study in study_iterator_global(study_classes=study_classes, only_first_one=only_first_one,
                                                          altitudes=altitudes)]
        study_classes_str = [get_display_name_from_object_type(c) for c in study_classes]
        quantity_altitude_tuples = list(product(study_classes_str, altitudes))
        quantity_altitude_to_visualizer = OrderedDict(zip(quantity_altitude_tuples, visualizers))
        visualizer = QuantityAltitudeHypercubeVisualizer(quantity_altitude_to_visualizer, save_to_file=save_to_file,
                                                         trend_test_class=trend_test_class, fast=fast)
        visualizer.visualize_year_trend_test(add_detailed_plots=add_detailed_plots)
        visualizer.visualize_massif_trend_test(add_detailed_plots=add_detailed_plots)
        visualizer.visualize_altitude_trend_test(add_detailed_plots=add_detailed_plots)


def fast_trends_with_altitude_hypercube():
    save_to_file = False
    only_first_one = False
    fast = True
    altitudes = ALL_ALTITUDES[2:4]
    for study_class in SCM_STUDIES[:1]:
        for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][1:2]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                           for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                     trend_test_class=trend_test_class, fast=fast)
            visualizer.visualize_year_trend_test()
            visualizer.visualize_massif_trend_test()
            visualizer.visualize_altitude_trend_test()


def fast_trends_with_quantity_altitude_hypercube():
    save_to_file = False
    only_first_one = False
    fast = True
    altitudes = ALL_ALTITUDES[2:4]
    study_classes = SCM_STUDIES[:2]
    for trend_test_class in [MannKendallTrendTest, GevLocationTrendTest, GevScaleTrendTest, GevShapeTrendTest][:1]:
        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                       for study in study_iterator_global(study_classes=study_classes, only_first_one=only_first_one,
                                                          altitudes=altitudes)]
        study_classes_str = [get_display_name_from_object_type(c) for c in study_classes]
        quantity_altitude_tuples = list(product(study_classes_str, altitudes))
        quantity_altitude_to_visualizer = OrderedDict(zip(quantity_altitude_tuples, visualizers))
        visualizer = QuantityAltitudeHypercubeVisualizer(quantity_altitude_to_visualizer, save_to_file=save_to_file,
                                                         trend_test_class=trend_test_class, fast=fast)
        visualizer.visualize_year_trend_test()
        # visualizer.visualize_massif_trend_test()
        # visualizer.visualize_altitude_trend_test()


def main_run():
    # fast_trends_with_altitude_hypercube()
    # fast_trends_with_quantity_altitude_hypercube()
    full_trends_with_quantity_altitude_hypercube()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
