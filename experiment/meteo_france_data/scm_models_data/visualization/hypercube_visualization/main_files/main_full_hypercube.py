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
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevLocationTrendTest


def get_full_parameters(altitude=None):
    save_to_file = True
    only_first_one = False
    nb_data_reduced_for_speed = False
    if altitude is not None:
        altitudes = [altitude]
    else:
        altitudes = ALL_ALTITUDES[3:-6]
    first_starting_year = 1958 + 10
    last_starting_year = 2017 - 10
    trend_test_class = GevLocationTrendTest
    return altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class


def get_full_altitude_visualizer(altitude_hypercube_class, study_classes, exact_starting_year=None, altitude=None):
    altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_full_parameters(altitude=altitude)
    if exact_starting_year is not None:
        last_starting_year = None
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class, exact_starting_year=exact_starting_year, first_starting_year=first_starting_year)
    return visualizer


def get_full_quantity_visualizer(quantity_hypercube_class, altitude=None, study_classes=None):
    altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_full_parameters(altitude=altitude)
    if study_classes is None:
        study_classes = SCM_STUDIES[:3]
    visualizer = load_quantity_visualizer(quantity_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one,
                                          save_to_file, study_classes, trend_test_class)
    return visualizer


def main_mean_log_likelihood():
    # Main plot
    get_full_quantity_visualizer(QuantityHypercubeWithoutTrend).visualize_year_trend_test(add_detailed_plots=True)
    # Detailed plot
    # get_full_quantity_visualizer(QuantityHypercubeWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes(
    #     add_detailed_plot=True)


    # get_full_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes()


def main_percentage_trend():
    for study_class in SCM_STUDIES:
        study_classees = [study_class]
        visualizer = get_full_altitude_visualizer(AltitudeHypercubeVisualizerBisExtended, exact_starting_year=1981,
                                                  study_classes=study_classees)
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
