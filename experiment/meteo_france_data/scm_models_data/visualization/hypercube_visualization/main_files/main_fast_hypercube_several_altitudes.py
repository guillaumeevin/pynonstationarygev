import time

from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer_extended import \
    AltitudeHypercubeVisualizerBisExtended, QuantityHypercubeWithoutTrendExtended, \
    AltitudeHypercubeVisualizerWithoutTrendExtended, QuantityHypercubeWithoutTrend
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer, load_quantity_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES, SCM_STUDIES
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest


def get_fast_parameters(altitude=None):
    save_to_file = False
    only_first_one = False
    nb_data_reduced_for_speed = 4
    if altitude is not None:
        altitudes = [altitude]
    else:
        altitudes = [ALL_ALTITUDES[3], ALL_ALTITUDES[-7]]
    last_starting_year = None
    trend_test_class = GevLocationChangePointTest
    return altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class


def get_fast_altitude_visualizer(altitude_hypercube_class):
    altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters()
    study_classes = SCM_STUDIES[:1]
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class)
    return visualizer


def get_fast_quantity_visualizer(quantity_hypercube_class):
    altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters()
    study_classes = SCM_STUDIES[:2]
    visualizer = load_quantity_visualizer(quantity_hypercube_class, altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one,
                                          save_to_file, study_classes, trend_test_class)
    return visualizer


def main_mean_log_likelihood():
    # Simply the main graph
    get_fast_quantity_visualizer(QuantityHypercubeWithoutTrend).visualize_year_trend_test(add_detailed_plots=True)
    # get_fast_quantity_visualizer(QuantityHypercubeWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes(
    #     add_detailed_plot=True)
    # get_fast_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes()


def main_percentage_trend():
    visualizer = get_fast_altitude_visualizer(AltitudeHypercubeVisualizerBisExtended)
    visualizer.vsualize_year_trend_by_regions_and_altitudes()
    visualizer.visualize_massif_trend_test_by_altitudes()
    visualizer.visualize_altitute_trend_test_by_regions()


def main_run():
    main_mean_log_likelihood()
    # main_percentage_trend()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
