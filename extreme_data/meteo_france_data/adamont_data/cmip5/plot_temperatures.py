from typing import Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_linestyle_from_scenario, \
    adamont_scenarios_real, AdamontScenario, scenario_to_str, get_gcm_list, rcp_scenarios
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp, \
    years_and_global_mean_temps
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import VisualizationParameters, \
    StudyVisualizer
from root_utils import VERSION_TIME


def main_plot_temperature_all(anomaly=True, spline=True):
    ax = plt.gca()
    for gcm in get_gcm_list(adamont_version=2)[:]:
        for scenario in rcp_scenarios[:]:
            label=gcm if scenario == rcp_scenarios[0] else None
            plot_temperature_for_rcp_gcm(ax, gcm, scenario, label=label, year_min=2005, year_max=2100, spline=spline, anomaly=anomaly)

    end_plot(anomaly, ax, spline)


def main_plot_temperature_with_spline_on_top(anomaly=True):
    spline = None
    for gcm in get_gcm_list(adamont_version=2)[:]:
        ax = plt.gca()
        # Plot the historical part in solid line (this part is the same between the different scenarios)
        linestyle = get_linestyle_from_scenario(AdamontScenario.histo)
        scenarios = rcp_scenarios
        for scenario in scenarios:
            label = gcm if scenario == scenarios[0] else None
            plot_temperature_for_rcp_gcm(ax, gcm, scenario, year_min=1850, year_max=2005, linestyle=linestyle,
                                         label=label, spline=spline, anomaly=anomaly)
            plot_temperature_for_rcp_gcm(ax, gcm, scenario, year_min=2005, year_max=2100, spline=spline, anomaly=anomaly)

        title = '{} {} of temperatures'.format(gcm, 'anomaly' if anomaly else 'mean')
        end_plot(anomaly, ax, spline, title)


def end_plot(anomaly, ax, spline, title=None):
    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label=scenario_to_str(s),
               linestyle=get_linestyle_from_scenario(s)) for s in adamont_scenarios_real
    ]
    ax2.legend(handles=legend_elements, loc='center left')
    ax2.set_yticks([])
    ax.legend(loc='upper left')
    ax.set_xlabel('Years')
    if spline is None:
        add_str = ' with and without spline'
    elif spline:
        add_str = ' with spline'
    else:
        add_str = ''
    add_str1 = 'anomaly of temperature' if anomaly else 'mean Temperature'
    ax.set_ylabel('Global {}{} (K)\n'
                  'mean temperature is taken on the year centered on the winter'.format(add_str1, add_str))
    if title is None:
        plt.show()
    else:
        filename = "{}/{}".format(VERSION_TIME, title)
        StudyVisualizer.savefig_in_results(filename, transparent=False)
    plt.close()


def plot_temperature_for_rcp_gcm(ax, gcm, scenario, year_min, year_max, linestyle=None,
                                 label=None, spline: Union[None, bool] = True, anomaly=True):
    splines = [spline] if spline is not None else [True, False]
    for spline in splines:
        years, global_mean_temp = years_and_global_mean_temps(gcm, scenario, year_min=year_min, year_max=year_max, spline=spline, anomaly=anomaly)
        if len(splines) == 2:
            if spline:
                color = 'k'
                label_plot = None if label is None else label + ' with spline'
            else:
                color = gcm_to_color[gcm]
                label_plot = None if label is None else label + ' without spline'
        else:
            color = gcm_to_color[gcm]
            label_plot = label
        if linestyle is None:
            linestyle = get_linestyle_from_scenario(scenario)
        ax.plot(years, global_mean_temp, linestyle=linestyle, color=color, label=label_plot)


if __name__ == '__main__':
    for anomaly in [True, False][:1]:
        main_plot_temperature_with_spline_on_top(anomaly=anomaly)
        # main_plot_temperature_all(anomaly=True, spline=True)
