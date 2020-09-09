from extreme_data.meteo_france_data.scm_models_data.visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

from projects.exceeding_snow_loads.section_discussion.crocus_study_comparison_with_eurocode import \
    CrocusDifferenceSnowLoad, \
    CrocusSnowDensityAtMaxofSwe, CrocusSnowDepthDifference
from projects.exceeding_snow_loads.utils import dpi_paper1_figure


def max_graph_annual_maxima_comparison():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = True
    study_classes = [
                        CrocusSnowDensityAtMaxofSwe,
                        CrocusDifferenceSnowLoad,
                        CrocusSnowDepthDifference,
                    ][:]
    study_class_to_ylim_and_yticks = {
        CrocusSnowDensityAtMaxofSwe: ([150, 500], [50 * i for i in range(3, 11)]),
        CrocusDifferenceSnowLoad: ([0, 10], [2 * i for i in range(0, 6)]),
        CrocusSnowDepthDifference: ([0, 0.6], [0.2 * i for i in range(0, 4)]),
    }
    for study_class in study_classes:
        ylim, yticks = study_class_to_ylim_and_yticks[study_class]

        # marker_altitude_massif_name = [
        #                                   ('magenta', 900, 'Ubaye'),
        #                                   ('darkmagenta', 1800, 'Vercors'),
        #                                   ('mediumpurple', 2700, 'Beaufortain'),
        #                               ][:]

        marker_altitude = [
                              ('magenta', 900),
                              ('darkmagenta', 1800),
                              ('mediumpurple', 2700),
                          ][:]
        ax = plt.gca()
        legend_size = 14

        for color, altitude in marker_altitude:
            for study in study_iterator_global([study_class], altitudes=[altitude]):
                study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                                   verbose=True,
                                                   multiprocessing=True)
                snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
                nb_massifs = len(study_visualizer.study.study_massif_names)
                label = 'Mean at {}m ({} massifs)'.format(altitude, nb_massifs)
                study_visualizer.visualize_max_graphs_poster(None, altitude, snow_abbreviation, color, label,
                                                             False, ax, legend_size=legend_size)

                last_plot = altitude == 2700
                if last_plot:
                    # if study_class == CrocusSnowDensityAtMaxofSwe:
                    #     label = '{} for French standards'.format(snow_density_str)
                    #     snow_density_eurocode = [150 for _ in study.ordered_years]
                    #     ax.plot(study.ordered_years, snow_density_eurocode, color='k', label=label)
                    ax.legend(prop={'size': legend_size})
                    tight_pad = {'h_pad': 0.2}
                    ax.set_ylim(ylim)
                    ax.set_xlim([1957, 2018])
                    ax.yaxis.set_ticks(yticks)
                    study_visualizer.show_or_save_to_file(no_title=True, tight_layout=True,
                                                          tight_pad=tight_pad, dpi=dpi_paper1_figure)
                    ax.clear()


if __name__ == '__main__':
    max_graph_annual_maxima_comparison()
