from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.altitude_hypercube_visualizer_extended import \
    QuantityHypercubeWithoutTrend, AltitudeHypercubeVisualizerBisExtended
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_fast_hypercube_one_altitudes import \
    get_fast_parameters, get_fast_quantity_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.main_files.main_full_hypercube import \
    get_full_quantity_visualizer, get_full_altitude_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.hypercube_visualization.utils_hypercube import \
    load_altitude_visualizer
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    SCM_STUDIES

POSTER_ALTITUDES = [900, 1800, 2700]


def fast_poster():
    for altitude in POSTER_ALTITUDES[:1]:
        study_classes = SCM_STUDIES[:2]
        results = get_fast_quantity_visualizer(QuantityHypercubeWithoutTrend,
                                               altitude=altitude,
                                               study_classes=study_classes).visualize_year_trend_test(add_detailed_plots=True)
        study_class_to_year = dict(zip(study_classes, [t[1] for t in results]))
        for study_class, exact_year in study_class_to_year.items():
            altitudes, last_starting_year, nb_data_reduced_for_speed, only_first_one, save_to_file, trend_test_class = get_fast_parameters(
                altitude=altitude)
            spatial_visualizer = load_altitude_visualizer(AltitudeHypercubeVisualizerBisExtended, altitudes,
                                                          last_starting_year,
                                                          nb_data_reduced_for_speed, only_first_one, save_to_file,
                                                          [study_class],
                                                          trend_test_class,
                                                          exact_starting_year=exact_year)
            spatial_visualizer.visualize_massif_trend_test_one_altitude()


def full_poster():
    for altitude in POSTER_ALTITUDES[:]:
        study_classes = SCM_STUDIES[:]
        results = get_full_quantity_visualizer(QuantityHypercubeWithoutTrend,
                                               altitude=altitude,
                                               study_classes=study_classes).visualize_year_trend_test(add_detailed_plots=True)
        study_class_to_year = dict(zip(study_classes, [t[1] for t in results]))
        for study_class, exact_year in study_class_to_year.items():
            spatial_visualizer = get_full_altitude_visualizer(AltitudeHypercubeVisualizerBisExtended, [study_class], exact_starting_year=exact_year, altitude=altitude)
            spatial_visualizer.visualize_massif_trend_test_one_altitude()


if __name__ == '__main__':
    full_poster()
