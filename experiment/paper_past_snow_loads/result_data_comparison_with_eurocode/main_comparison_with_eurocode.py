from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusDepth
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDepthVariable
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    study_iterator_global, SCM_STUDY_CLASS_TO_ABBREVIATION, snow_density_str
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
import matplotlib.pyplot as plt

from experiment.paper_past_snow_loads.result_data_comparison_with_eurocode.crocus_study_comparison_with_eurocode import \
    CrocusDifferenceSnowLoad, \
    CrocusSnowDensityAtMaxofSwe, CrocusDifferenceSnowLoadRescaledAndEurocodeToSeeSynchronization, \
    CrocusSnowDepthAtMaxofSwe, CrocusSnowDepthDifference


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
    for study_class in study_classes:

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
                    constant = 150 if study_class == CrocusSnowDensityAtMaxofSwe else 0
                    label = '{} Eurocode'.format(
                        snow_density_str) if study_class == CrocusSnowDensityAtMaxofSwe else None
                    snow_density_eurocode = [constant for _ in study.ordered_years]
                    ax.plot(study.ordered_years, snow_density_eurocode, color='k', label=label)
                    ax.legend()
                    tight_pad = {'h_pad': 0.2}
                    study_visualizer.show_or_save_to_file(no_title=True, tight_layout=True, tight_pad=tight_pad)
                    ax.clear()


if __name__ == '__main__':
    max_graph_annual_maxima_comparison()
