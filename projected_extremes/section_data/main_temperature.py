import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extreme_data.cru_data.global_mean_temperature_until_2020 import \
    winter_year_to_averaged_global_mean_temp_wrt_1850_1900
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_list
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import years_and_global_mean_temps
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from root_utils import VERSION_TIME


def main_plot_temperature_with_spline_on_top(anomaly=True):
    ax = plt.gca()
    scenario = AdamontScenario.rcp85
    splines = [True, False][:1]
    for gcm in get_gcm_list()[:]:
        for spline in splines:
            linestyle, linewidth, label = get_setting(spline, gcm + ' GCM')
            color = gcm_to_color[gcm]
            years, global_mean_temp = years_and_global_mean_temps(gcm, scenario, year_min=1850, year_max=2100,
                                                                  spline=spline, anomaly=anomaly)
            ax.plot(years, global_mean_temp, linestyle=linestyle, color=color, label=label, linewidth=linewidth)
    # plot observation
    for spline in splines:
        d = winter_year_to_averaged_global_mean_temp_wrt_1850_1900(spline)
        years, global_mean_temp = list(d.keys()), list(d.values())
        linestyle, linewidth, label = get_setting(spline, "HadCRUT5 reanalysis")
        linewidth *= 1
        ax.plot(years, global_mean_temp, linestyle=linestyle, color='k', label=label, linewidth=linewidth)

    if len(splines) == 2:
        ax2 = ax.twinx()
        legend_elements = [
            Line2D([0], [0], color='k', lw=1, label="Smoothed global mean", linestyle='-'),
            Line2D([0], [0], color='k', lw=1, label="Raw global mean", linestyle='dotted'),
        ]
        ax2.legend(handles=legend_elements, loc='center left')
        ax2.set_yticks([])

    title = 'Anomaly'
    ax.legend(loc='upper left', prop={'size': 14})
    ax.set_xlabel('Years', fontsize=13)
    ax.set_ylabel('Anomaly of global mean temperature\nwith respect to pre-industrial levels ($^o$C)', fontsize=13)
    ax.set_xlim((1850, 2100))
    ax.tick_params(axis='both', which='major', labelsize=13)
    # plt.show()
    if title is None:
        plt.show()
    else:
        filename = "{}/{}".format(VERSION_TIME, title)
        # StudyVisualizer.savefig_in_results(filename, transparent=False)
        StudyVisualizer.savefig_in_results(filename, transparent=True)
    plt.close()


def get_setting(spline, label):
    linestyle = '-' if spline else 'dotted'
    linewidth = 2 if spline else 1
    label = label if spline else None
    return linestyle, linewidth, label


if __name__ == '__main__':
    main_plot_temperature_with_spline_on_top()
