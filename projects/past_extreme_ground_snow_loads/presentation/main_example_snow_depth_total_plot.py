import matplotlib.pyplot as plt

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusDepth
from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
from projects.past_extreme_ground_snow_loads.utils import dpi_paper1_figure


def tuples_for_examples_paper1(examples_for_the_paper=True):
    if examples_for_the_paper:

        marker_altitude_massif_name_for_paper1 = [
            # ('magenta', 900, 'Ubaye'),
            ('darkblue', 900, 'Chartreuse'),
            # ('mediumpurple', 2700, 'Beaufortain'),
        ]
    else:
        marker_altitude_massif_name_for_paper1 = [
            ('magenta', 600, 'Parpaillon'),
            # ('darkmagenta', 300, 'Devoluy'),
            ('mediumpurple', 300, 'Aravis'),
        ]
    return marker_altitude_massif_name_for_paper1


def max_graph_annual_maxima_poster():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = True
    study_class = CrocusDepth

    examples_for_the_paper = True

    ax = plt.gca()
    if examples_for_the_paper:
        # ax.set_ylim([0, 20])
        # ax.set_yticks(list(range(0, 21, 2)))
        linewidth = 5
    else:
        linewidth = 3

    marker_altitude_massif_name_for_paper1 = tuples_for_examples_paper1(examples_for_the_paper)

    for color, altitude, massif_name in marker_altitude_massif_name_for_paper1[::-1]:
        for study in study_iterator_global([study_class], altitudes=[altitude]):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
            snow_abbreviation = 'snow depth'
            last_plot = True
            label = '{} massif at {}m'.format(massif_name, altitude)
            tight_pad = {'h_pad': 0.2}
            snow_abbreviation = 'annual maximum of ' + snow_abbreviation
            study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label,
                                                         last_plot, ax, tight_pad=tight_pad,
                                                         dpi=dpi_paper1_figure,
                                                         linewidth=linewidth)


if __name__ == '__main__':
    max_graph_annual_maxima_poster()
