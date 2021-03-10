import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import gcm_to_color
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, rcp_scenarios, get_gcm_list, \
    get_linestyle_from_scenario, scenario_to_str, adamont_scenarios_real
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import years_and_global_mean_temps


def get_year_min_and_year_max(gcm, scenario, left_limit, right_limit, is_temperature_interval):
    if is_temperature_interval:
        years_to_select = _get_year_min_and_year_max_for_temperature_interval(gcm, left_limit,
                                                                              right_limit, scenario)
        if len(years_to_select) == 0:
            return None, None
        year_min, year_max = years_to_select[0], years_to_select[-1]
    else:
        year_min, year_max = left_limit, right_limit

    # A minimum of 30 years of data is needed to find a trend
    if year_max - year_min + 1 >= 30:
        return year_min, year_max
    else:
        return None, None


def _get_year_min_and_year_max_for_temperature_interval(gcm, left_limits, right_limits, scenario):
    years, global_mean_temps = years_and_global_mean_temps(gcm, scenario, year_min=2006, year_max=2100, anomaly=True,
                                                           spline=True)
    years, global_mean_temps = np.array(years), np.array(global_mean_temps)
    ind = left_limits < global_mean_temps
    ind &= global_mean_temps < right_limits
    years_to_select = years[ind]
    ind2 = years_to_select[:-1] == years_to_select[1:] - 1
    if not all(ind2):
        i = list(ind2).index(False)
        years_to_select = years_to_select[:i + 1]
    return years_to_select


def get_nb_data(gcm, scenario, temperature_min, temperature_max, is_temperature_interval):
    year_min, year_max = get_year_min_and_year_max(gcm, scenario, temperature_min, temperature_max, is_temperature_interval)
    if year_min is None:
        return 0
    else:
        return year_max - year_min + 1


def plot_nb_data_one_line(ax, gcm, scenario, left_limits, right_limits, first_scenario, is_temperature_interval):
    nb_data = [get_nb_data(gcm, scenario, left, right, is_temperature_interval) for left, right in zip(left_limits, right_limits)]
    color = gcm_to_color[gcm]
    linestyle = get_linestyle_from_scenario(scenario)

    # Filter out the zero value
    nb_data, right_limits = np.array(nb_data), np.array(right_limits)
    ind = np.array(nb_data) > 0
    nb_data, right_limits = nb_data[ind], right_limits[ind]

    # For the legend
    if (len(nb_data) > 0) and first_scenario:
        ax.plot(right_limits[0], nb_data[0], color=color, linestyle='solid', label=gcm)

    ax.plot(right_limits, nb_data, linestyle=linestyle, color=color, marker='o')


def plot_nb_data(is_temperature_interval, is_shift_interval):
    left_limit, right_limit = get_interval_limits(is_temperature_interval, is_shift_interval)

    ax = plt.gca()
    for gcm in get_gcm_list(adamont_version=2)[:]:
        for i, scenario in enumerate(rcp_scenarios[2:]):
            plot_nb_data_one_line(ax, gcm, scenario, left_limit, right_limit,
                                  i == 0, is_temperature_interval)

    ax.legend()
    ticks_labels = get_ticks_labels_for_interval(is_temperature_interval, is_shift_interval)
    ax.set_xticks(right_limit)
    ax.set_xticklabels(ticks_labels)
    # ax.set_xlabel('Interval')
    ax.set_ylabel('Nb of Maxima')
    ax2 = ax.twinx()
    legend_elements = [
        Line2D([0], [0], color='k', lw=1, label=scenario_to_str(s),
               linestyle=get_linestyle_from_scenario(s)) for s in adamont_scenarios_real
    ]
    ax2.legend(handles=legend_elements, loc='upper center')
    ax2.set_yticks([])
    plt.show()


def get_interval_limits(is_temperature_interval, is_shift_interval):
    if is_temperature_interval:
        temp_min = np.arange(0, 3, 1)
        temp_max = temp_min + 2
        left_limit, right_limit = temp_min, temp_max
    else:
        shift = 25
        nb = 3
        year_min = [2006 + shift * i for i in range(nb)]
        year_max = [2050 + shift * i for i in range(nb)]
        left_limit, right_limit = year_min, year_max
    if not is_shift_interval:
        min_interval_left = min(left_limit)
        left_limit = [min_interval_left for _ in right_limit]
    return left_limit, right_limit


def get_ticks_labels_for_interval(is_temperature_interval, is_shift_interval):
    left_limits, right_limits = get_interval_limits(is_temperature_interval, is_shift_interval)
    ticks_labels = [' +${}^o\mathrm{C}$ and +${}^o\mathrm{C}$'.format(left_limit, right_limit, **{'C': '{C}'})
                    if is_temperature_interval else '{} and {}'.format(left_limit, right_limit)
                    for left_limit, right_limit in zip(left_limits, right_limits)]
    prefix = 'Maxima between \n'
    ticks_labels = [prefix + l for l in ticks_labels]
    return ticks_labels


if __name__ == '__main__':
    for shift_interval in [False, True]:
        for temp_interval in [False, True][1:]:
            print("shift = {}, temp_inteval = {}".format(shift_interval, temp_interval))
            plot_nb_data(is_temperature_interval=temp_interval, is_shift_interval=shift_interval)
