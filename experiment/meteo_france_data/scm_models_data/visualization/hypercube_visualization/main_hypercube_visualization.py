import time
from collections import OrderedDict
from itertools import product

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer_extended import \
    AltitudeYearHypercubeVisualizerExtended, AltitudeHypercubeVisualizerExtended, AltitudeHypercubeVisualizerBisExtended
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES, SCM_STUDIES, study_iterator, study_iterator_global
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest, \
    GevScaleChangePointTest, GevShapeChangePointTest
from utils import get_display_name_from_object_type


def full_trends_with_altitude_hypercube():
    save_to_file = True
    only_first_one = False
    fast = False
    altitudes = ALL_ALTITUDES[3:-6]
    for study_class in SCM_STUDIES[:1]:
        for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:1]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                           for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                     trend_test_class=trend_test_class, nb_data_reduced_for_speed=False)
            visualizer.visualize_massif_trend_test()
            visualizer.visualize_year_trend_test()
            visualizer.visualize_altitude_trend_test()


def full_quantity_altitude_hypercube():
    save_to_file = True
    only_first_one = False
    fast = False
    add_detailed_plots = True
    altitudes = ALL_ALTITUDES[3:-6]
    study_classes = SCM_STUDIES
    for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:]:
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


def fast_altitude_hypercube():
    save_to_file = False
    only_first_one = False
    fast = True
    altitudes = [ALL_ALTITUDES[3], ALL_ALTITUDES[-7]]
    for study_class in SCM_STUDIES[:1]:
        for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:1]:
            visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                           for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                       altitudes=altitudes)]
            altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
            visualizer = AltitudeHypercubeVisualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                     trend_test_class=trend_test_class, nb_data_reduced_for_speed=fast)
            # visualizer.visualize_year_trend_test()
            # visualizer.visualize_massif_trend_test()
            visualizer.visualize_altitude_trend_test()


def fast_altitude_year_hypercube():
    save_to_file = False
    only_first_one = False
    nb_data_reduced_for_speed = True
    altitudes = [ALL_ALTITUDES[3], ALL_ALTITUDES[-7]]
    for study_class in SCM_STUDIES[:1]:
        for last_starting_year in [None, 1989, 1999][:1]:
            for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:1]:
                visualizers = [
                    StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                    for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                altitudes=altitudes)]
                altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
                visualizer = Altitude_Hypercube_Year_Visualizer(altitude_to_visualizer, save_to_file=save_to_file,
                                                                trend_test_class=trend_test_class,
                                                                nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                                                last_starting_year=last_starting_year)
                # visualizer.visualize_year_trend_test()
                visualizer.visualize_altitude_trend_test()
                # visualizer.visualize_massif_trend_test()


def fast_altitude_year_hypercube_extended():
    save_to_file = True
    only_first_one = False
    nb_data_reduced_for_speed = True
    altitudes = [ALL_ALTITUDES[3], ALL_ALTITUDES[-7]]
    for study_class in SCM_STUDIES[:2]:
        for last_starting_year in [None, 1989, 1999][:2]:
            for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:2]:
                visualizers = [
                    StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                    for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                altitudes=altitudes)]
                altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
                visualizer = AltitudeHypercubeVisualizerExtended(altitude_to_visualizer, save_to_file=save_to_file,
                                                                 trend_test_class=trend_test_class,
                                                                 nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                                                 last_starting_year=last_starting_year)
                # visualizer.visualize_year_trend_test()
                visualizer.vsualize_year_trend_by_regions_and_altitudes()
                # visualizer.visualize_altitude_trend_test()
                # visualizer.visualize_massif_trend_test_by_altitudes()
                visualizer.visualize_massif_trend_test_by_altitudes()
                visualizer.visualize_altitute_trend_test_by_regions()
                # visualizer.visualize_massif_trend_test()


def full_altitude_year_hypercube():
    save_to_file = True
    only_first_one = False
    nb_data_reduced_for_speed = False
    altitudes = ALL_ALTITUDES[3:-6]
    for study_class in SCM_STUDIES[:1]:
        for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest,
                                 GevShapeChangePointTest][:1]:
            years = [1967, 1977, 1987, 1997, 2007, None][-2:-1][::-1]
            for last_starting_year in years:
                visualizers = [
                    StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                    for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                altitudes=altitudes)]
                altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
                visualizer = Altitude_Hypercube_Year_Visualizer(altitude_to_visualizer,
                                                                save_to_file=save_to_file,
                                                                trend_test_class=trend_test_class,
                                                                nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                                                last_starting_year=last_starting_year)
                visualizer.visualize_year_trend_test()
                visualizer.visualize_massif_trend_test()
                visualizer.visualize_altitude_trend_test()


def full_altitude_year_hypercube_extended():
    save_to_file = True
    only_first_one = False
    nb_data_reduced_for_speed = False
    altitudes = ALL_ALTITUDES[3:-6]
    for study_class in SCM_STUDIES[:]:
        for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest,
                                 GevShapeChangePointTest][:1]:
            years = [1967, 1977, 1987, 1997, 2007, None][-4:][::-1]
            for last_starting_year in years:
                for days in [1, 3][1:]:
                    visualizers = [
                        StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                        for study in study_iterator(study_class=study_class, only_first_one=only_first_one,
                                                    altitudes=altitudes, nb_consecutive_days=days)]
                    altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
                    visualizer = AltitudeHypercubeVisualizerBisExtended(altitude_to_visualizer,
                                                                        save_to_file=save_to_file,
                                                                        trend_test_class=trend_test_class,
                                                                        nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                                                        last_starting_year=last_starting_year,
                                                                        )
                    visualizer.visualize_altitute_trend_test_by_regions()
                    visualizer.visualize_massif_trend_test_by_altitudes()
                    visualizer.vsualize_year_trend_by_regions_and_altitudes()
                    # visualizer.visualize_year_trend_test()
                    # visualizer.visualize_massif_trend_test()
                    # visualizer.visualize_altitude_trend_test()


def fast_quantity_altitude_hypercube():
    save_to_file = False
    only_first_one = False
    fast = True
    altitudes = ALL_ALTITUDES[2:4]
    study_classes = SCM_STUDIES[:2]
    for trend_test_class in [GevLocationChangePointTest, GevScaleChangePointTest, GevShapeChangePointTest][:1]:
        visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                       for study in study_iterator_global(study_classes=study_classes, only_first_one=only_first_one,
                                                          altitudes=altitudes)]
        study_classes_str = [get_display_name_from_object_type(c) for c in study_classes]
        quantity_altitude_tuples = list(product(study_classes_str, altitudes))
        quantity_altitude_to_visualizer = OrderedDict(zip(quantity_altitude_tuples, visualizers))
        visualizer = QuantityAltitudeHypercubeVisualizer(quantity_altitude_to_visualizer, save_to_file=save_to_file,
                                                         trend_test_class=trend_test_class, fast=fast)
        visualizer.visualize_year_trend_test()
        visualizer.visualize_massif_trend_test()
        visualizer.visualize_altitude_trend_test()


def main_run():
    # fast_altitude_hypercube()
    # fast_altitude_year_hypercube()

    # fast_altitude_year_hypercube_extended()
    full_altitude_year_hypercube_extended()

    # full_altitude_year_hypercube()
    # fast_quantity_altitude_hypercube()
    # full_quantity_altitude_hypercube()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
