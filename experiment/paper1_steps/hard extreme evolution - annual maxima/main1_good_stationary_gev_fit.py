from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusRecentSwe
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer


def maxima_analysis():
    save_to_file = False
    only_first_one = False
    durand_altitude = [900, 1500, 1800, 2100, 2700][2:-2]
    altitudes = durand_altitude
    study_classes = [CrocusRecentSwe][:]
    for study in study_iterator_global(study_classes, only_first_one=only_first_one, altitudes=altitudes):
        study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                           verbose=True,
                                           multiprocessing=True)
        # study_visualizer.visualize_summary_of_annual_values_and_stationary_gev_fit()
        study_visualizer.visualize_all_mean_and_max_graphs()


if __name__ == '__main__':
    maxima_analysis()
