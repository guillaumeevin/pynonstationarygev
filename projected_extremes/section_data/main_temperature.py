from typing import Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extreme_data.cru_data.global_mean_temperature_until_2020 import \
    winter_year_to_averaged_global_mean_temp_wrt_1850_1900
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_linestyle_from_scenario, \
    adamont_scenarios_real, AdamontScenario, scenario_to_str, get_gcm_list, rcp_scenarios
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp, \
    years_and_global_mean_temps
from extreme_data.meteo_france_data.adamont_data.cmip5.plot_temperatures import plot_temperature_for_rcp_gcm
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import VisualizationParameters, \
    StudyVisualizer
from root_utils import VERSION_TIME

def main_plot_temperature_with_spline_on_top(anomaly=True):
    ax = plt.gca()
    scenario = AdamontScenario.rcp85
    for gcm in get_gcm_list(adamont_version=2)[:]:
        for spline in [True, False]:
            linestyle, linewidth, label = get_setting(spline, gcm  + ' GCM')
            color = gcm_to_color[gcm]
            years, global_mean_temp = years_and_global_mean_temps(gcm, scenario, year_min=1850, year_max=2100,
                                                                  spline=spline, anomaly=anomaly)
            ax.plot(years, global_mean_temp, linestyle=linestyle, color=color, label=label, linewidth=linewidth)
    # plot observation
    for spline in [True, False]:
        d = winter_year_to_averaged_global_mean_temp_wrt_1850_1900(spline)
        years, global_mean_temp = list(d.keys()), list(d.values())
        linestyle, linewidth, label = get_setting(spline, "HadCRUT5 reanalysis")
        linewidth *= 1
        ax.plot(years, global_mean_temp, linestyle=linestyle, color='k', label=label, linewidth=linewidth)

    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label="Smoothed global mean", linestyle='-'),
        Line2D([0], [0], color='k', lw=1, label="Raw global mean", linestyle='dotted'),
    ]
    ax2.legend(handles=legend_elements, loc='center left')
    ax2.set_yticks([])

    title = 'Anomaly'
    ax.legend(loc='upper left')
    ax.set_xlabel('Years')
    ax.set_ylabel('Anomaly of global mean temperature\nwith respect to pre-industrial levels ($^o$C)')
    ax.set_xlim((1850, 2100))
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
