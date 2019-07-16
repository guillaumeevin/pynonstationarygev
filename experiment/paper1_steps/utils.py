import time

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusRecentSwe
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_year_hypercube_visualizer import \
    Altitude_Hypercube_Year_Visualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_full_hypercube import \
    get_full_parameters
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevScaleChangePointTest, \
    GevLocationChangePointTest

FULL_ALTITUDES = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]


def get_full_altitude_visualizer(altitude_hypercube_class, exact_starting_year=None, altitude=900,
                                           reduce_strength_array=False):
    altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_full_parameters(
        altitude=altitude)
    if exact_starting_year is not None:
        first_starting_year, last_starting_year = None, None
    study_classes = [CrocusRecentSwe]
    trend_test_class = GevLocationChangePointTest
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class, first_starting_year=first_starting_year,
                                          exact_starting_year=exact_starting_year)
    visualizer.reduce_strength_array = reduce_strength_array
    return visualizer
