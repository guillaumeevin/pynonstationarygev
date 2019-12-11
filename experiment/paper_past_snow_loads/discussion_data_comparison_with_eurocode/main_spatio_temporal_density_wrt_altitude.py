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


def density_wrt_altitude():
    """
    We choose these massif because each represents a different eurocode region
    we also choose them because they belong to a different climatic area
    :return:
    """
    save_to_file = False
    study_class = CrocusSnowDensityAtMaxofSwe
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700][:-1]

    for spatial_plot in [False, True][::-1]:

        if spatial_plot:
            marker_lockeys_axis = [
                                      ('magenta', 'Ubaye'),
                                      ('darkmagenta', 'Vercors'),
                                      ('mediumpurple', 'Beaufortain'),
                                  ][:]
        else:
            marker_lockeys_axis = [
                                      ('magenta', [1958, 1987]),
                                      ('darkmagenta', [1968, 1997]),
                                      ('mediumpurple', [1978, 2007]),
                                      ('blue', [1988, 2017]),
                                  ][:]

        ax = plt.gca()

        j_to_mean_densities = {
            i: [] for i in range(len(marker_lockeys_axis))
        }
        for study in study_iterator_global([study_class], altitudes=altitudes):
            study_visualizer = StudyVisualizer(study, save_to_file=save_to_file,
                                               verbose=True,
                                               multiprocessing=True)
            for j, (color, lockeys) in enumerate(marker_lockeys_axis):
                if spatial_plot:
                    mean_density = study.observations_annual_maxima.df_maxima_gev.loc[lockeys, :].mean()
                else:
                    mean_density = study.observations_annual_maxima.df_maxima_gev.loc[:, lockeys[0]:lockeys[1]].mean().mean()
                j_to_mean_densities[j].append(mean_density)

        for j, (color, lockeys) in enumerate(marker_lockeys_axis):
            mean_densities = j_to_mean_densities[j]
            if spatial_plot:
                label = lockeys
            else:
                label = '-'.join([str(e) for e in lockeys])
            ax.plot(altitudes, mean_densities, color=color, label=label)

        snow_abbreviation = SCM_STUDY_CLASS_TO_ABBREVIATION[study_class]
        ax.legend()
        tight_pad = {'h_pad': 0.2}
        # ax.set_ylim(ylim)
        # ax.set_xlim([1957, 2018])
        ax.xaxis.set_ticks(altitudes)
        # ax.yaxis.set_ticks(yticks)
        study_visualizer.plot_name = ''
        study_visualizer.show_or_save_to_file(no_title=True, tight_layout=True,
                                              tight_pad=tight_pad, dpi=dpi_paper1_figure)
        ax.clear()


if __name__ == '__main__':
    density_wrt_altitude()
