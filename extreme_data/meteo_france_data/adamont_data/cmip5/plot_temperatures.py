import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import get_linestyle_from_scenario, \
    adamont_scenarios_real, AdamontScenario, scenario_to_str, get_gcm_list
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_global_mean_temp, \
    years_and_global_mean_temps


def main_plot_temperature():
    rolling = 30
    ax = plt.gca()
    for gcm in get_gcm_list(adamont_version=2)[:]:
        # Plot the historical part in solid line (this part is the same between the different scenarios)
        linestyle = get_linestyle_from_scenario(AdamontScenario.histo)
        plot_temperature_for_rcp_gcm(ax, gcm, AdamontScenario.rcp45, year_min=1951, year_max=2005, linestyle=linestyle,
                                     label=gcm, rolling=rolling)
        for scenario in adamont_scenarios_real[1:]:
            plot_temperature_for_rcp_gcm(ax, gcm, scenario, year_min=2005, year_max=2100, rolling=rolling)

    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label=scenario_to_str(s),
               linestyle=get_linestyle_from_scenario(s)) for s in adamont_scenarios_real
    ]
    ax2.legend(handles=legend_elements, loc='upper center')
    ax2.set_yticks([])


    ax.legend(loc='upper left')
    ax.set_xlabel('Years')
    add_str = ' averaged on the last {} years'.format(rolling) if rolling is not None else ''
    ax.set_ylabel('Global mean Temperature{} (K)\n'
                  'mean is taken on the year centered on the winter'.format(add_str))
    plt.show()


def plot_temperature_for_rcp_gcm(ax, gcm, scenario, year_min, year_max, linestyle=None, label=None, rolling=None):
    years, global_mean_temp = years_and_global_mean_temps(gcm, scenario, year_min=year_min, year_max=year_max, rolling=rolling)
    color = gcm_to_color[gcm]
    if linestyle is None:
        linestyle = get_linestyle_from_scenario(scenario)
    ax.plot(years, global_mean_temp, linestyle=linestyle, color=color, label=label)


if __name__ == '__main__':
    main_plot_temperature()
