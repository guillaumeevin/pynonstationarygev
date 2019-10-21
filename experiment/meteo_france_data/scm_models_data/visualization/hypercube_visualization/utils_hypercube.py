from collections import OrderedDict
from itertools import product

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from root_utils import get_display_name_from_object_type


def load_quantity_visualizer(quantity_hypercube_class, altitudes, last_starting_year, nb_data_reduced_for_speed,
                             only_first_one,
                             save_to_file, study_classes, trend_test_class):
    visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                   for study in study_iterator_global(study_classes=study_classes, only_first_one=only_first_one,
                                                      altitudes=altitudes)]
    study_classes_str = [get_display_name_from_object_type(c) for c in study_classes]
    quantity_altitude_tuples = list(product(study_classes_str, altitudes))
    quantity_altitude_to_visualizer = OrderedDict(zip(quantity_altitude_tuples, visualizers))
    visualizer = quantity_hypercube_class(quantity_altitude_to_visualizer,
                                          save_to_file=save_to_file,
                                          trend_test_class=trend_test_class,
                                          nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                          last_starting_year=last_starting_year)
    assert isinstance(visualizer, QuantityAltitudeHypercubeVisualizer)
    return visualizer


def load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year, nb_data_reduced_for_speed,
                             only_first_one, save_to_file, study_classes, trend_test_class
                             , exact_starting_year=None, first_starting_year=1958,
                             orientations=None,
                             verbose=True):
    visualizers = [StudyVisualizer(study, temporal_non_stationarity=True, verbose=False, multiprocessing=True)
                   for study in study_iterator_global(study_classes=study_classes, only_first_one=only_first_one,
                                                      altitudes=altitudes,
                                                      orientations=orientations,
                                                      verbose=verbose)]
    altitude_to_visualizer = OrderedDict(zip(altitudes, visualizers))
    visualizer = altitude_hypercube_class(altitude_to_visualizer,
                                          save_to_file=save_to_file,
                                          trend_test_class=trend_test_class,
                                          nb_data_reduced_for_speed=nb_data_reduced_for_speed,
                                          last_starting_year=last_starting_year,
                                          first_starting_year=first_starting_year,
                                          exact_starting_year=exact_starting_year,
                                          verbose=verbose,
                                          )
    assert isinstance(visualizer, AltitudeHypercubeVisualizer)
    return visualizer
