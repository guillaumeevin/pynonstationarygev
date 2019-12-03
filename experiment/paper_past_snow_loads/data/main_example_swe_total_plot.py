import matplotlib.pyplot as plt

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer


def max_graph_annual_maxima_poster():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = True
    study_class = CrocusSnowLoadTotal
    marker_altitude_massif_name = [
        ('magenta', 900, 'Ubaye'),
        ('darkmagenta', 1800, 'Vercors'),
        ('mediumpurple', 2700, 'Beaufortain'),
    ]
    ax = plt.gca()
    ax.set_ylim([0, 20])
    ax.set_yticks(list(range(0, 21, 2)))
    for color, altitude, massif_name in marker_altitude_massif_name:
        for study in study_iterator_global([study_class], altitudes=[altitude]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
            last_plot = altitude == 2700
            label = '{} massif at {}m'.format(massif_name, altitude)
            tight_pad = {'h_pad': 0.2}
            study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label,
                                                         last_plot, ax, tight_pad=tight_pad,
                                                         dpi=1000)


if __name__ == '__main__':
    max_graph_annual_maxima_poster()
