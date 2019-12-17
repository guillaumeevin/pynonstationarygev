import time

from experiment.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer_extended import \
    AltitudeHypercubeVisualizerBisExtended, QuantityHypercubeWithoutTrend
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer, load_quantity_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDIES
from experiment.trend_analysis.univariate_test.extreme_trend_test.trend_test_one_parameter import GevLocationTrendTest


def get_fast_parameters(altitude=1800):
    save_to_file = False
    only_first_one = False
    nb_data_reduced_for_speed = 4
    altitudes = [altitude]
    last_starting_year = None
    trend_test_class = GevLocationTrendTest
    return altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class


def get_fast_altitude_visualizer(altitude_hypercube_class, altitude=1800, study_class=SafranSnowfall, exact_year=None):
    altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters(altitude=altitude)
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, [study_class],
                                          trend_test_class, exact_starting_year=exact_year)
    return visualizer


def get_fast_quantity_visualizer(quantity_hypercube_class, altitude=1800, study_classes=None):
    altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters(altitude=altitude)
    if study_classes is None:
        study_classes = SCM_STUDIES[:2]
    visualizer = load_quantity_visualizer(quantity_hypercube_class, altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one,
                                          save_to_file, study_classes, trend_test_class)
    return visualizer


def main_mean_log_likelihood_poster_1():
    # Simply the main graph
    res = get_fast_quantity_visualizer(QuantityHypercubeWithoutTrend).visualize_year_trend_test(add_detailed_plots=True, poster_plot=True)
    # get_fast_quantity_visualizer(QuantityHypercubeWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes(
    #     add_detailed_plot=True)
    # get_fast_altitude_visualizer(AltitudeHypercubeVisualizerWithoutTrendExtended).vsualize_year_trend_by_regions_and_altitudes()


def main_percentage_trend_poster_2():
    visualizer = get_fast_altitude_visualizer(AltitudeHypercubeVisualizerBisExtended, exact_year=1958)
    # visualizer.vsualize_year_trend_by_regions_and_altitudes()
    # visualizer.visualize_massif_trend_test_by_altitudes()
    visualizer.visualize_massif_trend_test_one_altitude()
    # visualizer.visualize_altitute_trend_test_by_regions()


def main_run():
    main_mean_log_likelihood_poster_1()
    # main_percentage_trend_poster_2()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
