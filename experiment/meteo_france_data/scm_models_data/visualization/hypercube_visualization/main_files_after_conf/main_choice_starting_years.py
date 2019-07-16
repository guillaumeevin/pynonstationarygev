import time

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusRecentSwe
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer import \
    AltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer_extended import \
    AltitudeHypercubeVisualizerBisExtended, QuantityHypercubeWithoutTrendExtended, \
    AltitudeHypercubeVisualizerWithoutTrendExtended, QuantityHypercubeWithoutTrend
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_fast_hypercube_one_altitudes import \
    get_fast_parameters
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_full_hypercube import \
    get_full_parameters
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.quantity_altitude_visualizer import \
    QuantityAltitudeHypercubeVisualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer, load_quantity_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES, SCM_STUDIES
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest, \
    GevScaleChangePointTest


def get_fast_altitude_visualizer(altitude_hypercube_class):
    altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters()
    study_classes = [CrocusRecentSwe]
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class)
    return visualizer


def main_fast_old_spatial_repartition():
    # Simply the main graph
    get_fast_altitude_visualizer(Altitude_Hypercube_Year_Visualizer).visualize_massif_trend_test_one_altitude()


def get_full_altitude_visualizer(altitude_hypercube_class, exact_starting_year=None, altitude=900):
    altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_full_parameters(
        altitude=altitude)
    if exact_starting_year is not None:
        first_starting_year, last_starting_year = None, None
    study_classes = [CrocusRecentSwe]
    trend_test_class = GevScaleChangePointTest
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class, first_starting_year=first_starting_year,
                                          exact_starting_year=exact_starting_year)
    return visualizer


FULL_ALTITUDES = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
HALF_ALTITUDES = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000][::2]


def main_fast_spatial_repartition():
    for altitude in FULL_ALTITUDES[-1:]:
        vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                  exact_starting_year=1958)
        vizualiser.save_to_file = False
        vizualiser.visualize_massif_trend_test_one_altitude()


def main_full_spatial_repartition():
    for altitude in FULL_ALTITUDES[:]:
        # Compute for the most likely starting year
        # vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude)
        # vizualiser.visualize_massif_trend_test_one_altitude()
        # Compute the trend for a linear trend
        vizualiser = get_full_altitude_visualizer(Altitude_Hypercube_Year_Visualizer, altitude=altitude,
                                                  exact_starting_year=1958)
        vizualiser.visualize_massif_trend_test_one_altitude()


def main_run():
    main_full_spatial_repartition()
    # main_fast_spatial_repartition()


if __name__ == '__main__':
    start = time.time()
    main_run()
    duration = time.time() - start
    print('Full run took {}s'.format(round(duration, 1)))
