from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoad3Days, CrocusSnowLoadTotal, \
    CrocusSnowLoadEurocode
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

def max_graph_annual_maxima_poster_separate():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = False
    study_class = [CrocusSnowLoadTotal, CrocusSnowLoadEurocode][-1]
    marker_altitude_massif_name = [
        ('magenta', 900, 'Ubaye'),
        ('darkmagenta', 1800, 'Vercors'),
        ('mediumpurple', 2700, 'Beaufortain'),
    ]
    ax = plt.gca()
    for color, altitude, massif_name in marker_altitude_massif_name:
        for study in study_iterator_global([study_class], altitudes=[altitude]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
            last_plot = altitude == 2700
            label = '{} massif at {}m'.format(massif_name, altitude)
            study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label, last_plot, ax)


def max_graph_annual_maxima_poster_together():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = False
    study_class_and_marker = [
        (CrocusSnowLoadTotal, '-'),
        (CrocusSnowLoadEurocode, (0, (1,1))),
    ]
    color_altitude_massif_name = [
        ('magenta', 900, 'Ubaye'),
        ('darkmagenta', 1800, 'Vercors'),
        ('mediumpurple', 2700, 'Beaufortain'),
    ]
    ax = plt.gca()
    for color, altitude, massif_name in color_altitude_massif_name:
        for study_class, linestyle in study_class_and_marker[::-1]:
            for study in study_iterator_global([study_class], altitudes=[altitude]):
                study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                                   verbose=True,
                                                   multiprocessing=True)
                snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
                last_plot = altitude == 2700 and study_class == CrocusSnowLoadTotal
                name = 'SL from Crocus SWE' if study_class == CrocusSnowLoadTotal else 'SL from Crocus HS and snow density=150 kg $m^-3$'
                label = '{} for {} massif at {}m'.format(name, massif_name, altitude)
                study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label, last_plot, ax, linestyle)


if __name__ == '__main__':
    # max_graph_annual_maxima_poster_separate()
    max_graph_annual_maxima_poster_together()
