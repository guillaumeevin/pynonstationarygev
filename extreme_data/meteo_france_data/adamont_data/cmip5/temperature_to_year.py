import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, rcp_scenarios, get_gcm_list, \
    get_linestyle_from_scenario, scenario_to_str, adamont_scenarios_real
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import years_and_global_mean_temps


def temperature_minmax_to_year_minmax(gcm, scenario, temperature_min, temperature_max):
    years, global_mean_temps = years_and_global_mean_temps(gcm, scenario, year_min=2005, year_max=2100,
                                                           rolling=30, anomaly=True)
    years, global_mean_temps = np.array(years), np.array(global_mean_temps)
    ind = temperature_min < global_mean_temps
    ind &= global_mean_temps < temperature_max
    years_to_select = years[ind]
    ind2 = years_to_select[:-1] == years_to_select[1:] - 1
    if not all(ind2):
        i = list(ind2).index(False)
        years_to_select = years_to_select[:i + 1]
    # A minimum of 30 years of data is needed to find a trend
    if len(years_to_select) >= 30:
        year_min, year_max = years_to_select[0], years_to_select[-1]
        assert (year_max - year_min + 1) == len(years_to_select)
        return year_min, year_max
    else:
        return None, None


def get_nb_data(gcm, scenario, temperature_min, temperature_max):
    year_min, year_max = temperature_minmax_to_year_minmax(gcm, scenario, temperature_min, temperature_max)
    if year_min is None:
        return 0
    else:
        return year_max - year_min + 1


def plot_nb_data_one_line(ax, gcm, scenario, temp_min, temp_max, first_scenario):

    nb_data = [get_nb_data(gcm, scenario, mi, ma) for mi, ma in zip(temp_min, temp_max)]
    color = gcm_to_color[gcm]
    linestyle = get_linestyle_from_scenario(scenario)

    # Filter out the zero value
    nb_data, temp_min = np.array(nb_data), np.array(temp_min)
    ind = np.array(nb_data) > 0
    nb_data, temp_min = nb_data[ind], temp_min[ind]

    # For the legend
    if (len(nb_data) > 0) and first_scenario:
        ax.plot(temp_min[0], nb_data[0], color=color, linestyle='solid', label=gcm)

    ax.plot(temp_min, nb_data, linestyle=linestyle, color=color, marker='o')


def plot_nb_data():
    temp_min, temp_max = get_temp_min_and_temp_max()

    ax = plt.gca()
    for gcm in get_gcm_list(adamont_version=2)[:]:
        for i, scenario in enumerate(rcp_scenarios[:2]):
            plot_nb_data_one_line(ax, gcm, scenario, temp_min, temp_max, first_scenario=i == 0)

    ax.legend()
    ticks_labels = get_ticks_labels_for_temp_min_and_temp_max()
    ax.set_xticks(temp_min)
    ax.set_xticklabels(ticks_labels)
    ax.set_xlabel('Temperature interval')
    ax.set_ylabel('Nb of Maxima')
    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label=scenario_to_str(s),
               linestyle=get_linestyle_from_scenario(s)) for s in adamont_scenarios_real
    ]
    ax2.legend(handles=legend_elements, loc='upper center')
    ax2.set_yticks([])
    plt.show()


def get_temp_min_and_temp_max():
    temp_min = np.arange(0, 3, 1)
    temp_max = temp_min + 2
    return temp_min, temp_max

def get_ticks_labels_for_temp_min_and_temp_max():
    temp_min, temp_max = get_temp_min_and_temp_max()
    return ['Maxima occured between \n' \
            ' +${}^o\mathrm{C}$ and +${}^o\mathrm{C}$'.format(mi, ma, **{'C': '{C}'})
     for mi, ma in zip(temp_min, temp_max)]

if __name__ == '__main__':
    plot_nb_data()
