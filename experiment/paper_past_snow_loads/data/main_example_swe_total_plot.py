from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

def max_graph_annual_maxima_poster():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = False

    # marker_altitude_massif_name_and_study_class = [
    #     ('magenta', 900, 'Parpaillon', CrocusSnowLoadTotal),
    #     ('darkmagenta', 1800, 'Vercors', CrocusSnowLoadTotal),
    #     ('mediumpurple', 2700, 'Vanoise', CrocusSnowLoadTotal),
    # ]
    marker_altitude_massif_name_and_study_class = [
        ('magenta', 900, 'Ubaye', CrocusSnowLoadTotal),
        ('darkmagenta', 1800, 'Vercors', CrocusSnowLoadTotal),
        ('mediumpurple', 2700, 'Beaufortain', CrocusSnowLoadTotal),
    ]
    ax = plt.gca()
    for color, altitude, massif_name, study_class in marker_altitude_massif_name_and_study_class:
        for study in study_iterator_global([study_class], altitudes=[altitude]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
            last_plot = altitude == 2700
            label = '{} massif at {}m'.format(massif_name, altitude)
            study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label, last_plot, ax)

if __name__ == '__main__':
    max_graph_annual_maxima_poster()
