from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSwe3Days
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_full_hypercube import \
    get_full_parameters
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer
from experiment.trend_analysis.univariate_test.gev_trend_test_one_parameter import GevLocationTrendTest

FULL_ALTITUDES = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]


def get_full_altitude_visualizer(altitude_hypercube_class, exact_starting_year=None, altitude=900,
                                 reduce_strength_array=False,
                                 trend_test_class = GevLocationTrendTest,
                                 offset_starting_year=10,
                                 study_class=CrocusSwe3Days,
                                 orientation=None):
    altitudes, first_starting_year, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, _ = get_full_parameters(
        altitude=altitude, offset_starting_year=offset_starting_year)
    if exact_starting_year is not None:
        first_starting_year, last_starting_year = None, None
    study_classes = [study_class]
    visualizer = load_altitude_visualizer(altitude_hypercube_class, altitudes, last_starting_year,
                                          nb_data_reduced_for_speed, only_first_one, save_to_file, study_classes,
                                          trend_test_class, first_starting_year=first_starting_year,
                                          exact_starting_year=exact_starting_year,
                                          orientations=[orientation])
    visualizer.reduce_strength_array = reduce_strength_array
    return visualizer
