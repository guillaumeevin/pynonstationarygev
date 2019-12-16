from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDepthVariable
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION, snow_density_str
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

from experiment.paper_past_snow_loads.discussion_data_comparison_with_eurocode.crocus_study_comparison_with_eurocode import \
    CrocusDifferenceSnowLoad, \
    CrocusSnowDensityAtMaxofSwe, CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization, \
    CrocusSnowDepthAtMaxofSwe, CrocusSnowDepthDifference
from experiment.paper_past_snow_loads.paper_utils import dpi_paper1_figure


def max_graph_annual_maxima_comparison():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = True
    study_classes = [CrocusSnowDensityAtMaxofSwe,
                     # CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization,
                     CrocusDifferenceSnowLoad,
                     # CrocusDepth,
                     # CrocusSnowDepthAtMaxofSwe,
                     CrocusSnowDepthDifference,
                     ][:]
    study_class_to_ylim_and_yticks = {
        CrocusSnowDensityAtMaxofSwe: ([100, 500], [50*i for i in range(2, 11)]),
        CrocusDifferenceSnowLoad: ([0, 12], [2*i for i in range(0, 7)]),
        CrocusSnowDepthDifference: ([0, 1], [0.2*i for i in range(0, 6)]),
    }
    for study_class in study_classes:
        ylim, yticks = study_class_to_ylim_and_yticks[study_class]

        marker_altitude_massif_name = [
            ('magenta', 900, 'Ubaye'),
            ('darkmagenta', 1800, 'Vercors'),
            ('mediumpurple', 2700, 'Beaufortain'),
        ][:]
        ax = plt.gca()
        for color, altitude, massif_name in marker_altitude_massif_name:
            for study in study_iterator_global([study_class], altitudes=[altitude]):
                study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                                   verbose=True,
                                                   multiprocessing=True)
                snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
                label = '{} massif at {}m'.format(massif_name, altitude)
                study_visualizer.visualize_max_graphs_poster(massif_name, altitude, snow_abbreviation, color, label,
                                                             False, ax)
                last_plot = altitude == 2700
                if last_plot:
                    if study_class == CrocusSnowDensityAtMaxofSwe:
                        label = '{} for French standards'.format(snow_density_str)
                        snow_density_eurocode = [150 for _ in study.ordered_years]
                        ax.plot(study.ordered_years, snow_density_eurocode, color='k', label=label)
                    ax.legend()
                    tight_pad = {'h_pad': 0.2}
                    ax.set_ylim(ylim)
                    ax.set_xlim([1957, 2018])
                    ax.yaxis.set_ticks(yticks)
                    study_visualizer.show_or_save_to_file(no_title=True, tight_layout=True,
                                                          tight_pad=tight_pad, dpi=dpi_paper1_figure)
                    ax.clear()


if __name__ == '__main__':
    max_graph_annual_maxima_comparison()
