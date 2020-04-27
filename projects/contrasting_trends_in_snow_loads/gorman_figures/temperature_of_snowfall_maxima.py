import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.utils import SeasonForTheMaxima, season_to_str

from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranTemperature, \
    SafranPrecipitation1Day
from projects.contrasting_trends_in_snow_loads.gorman_figures.daily_snowfall_fraction import \
    compute_daily_snowfall_fraction


def distribution_temperature_of_maxima_of_snowfall(altitudes, temperature_at_maxima=True):
    season = SeasonForTheMaxima.annual if temperature_at_maxima else SeasonForTheMaxima.winter_extended
    # Load the temperature corresponding to the maxima of snowfall
    altitude_to_maxima_temperature = OrderedDict()
    altitude_to_optimal_temperature = OrderedDict()
    for altitude in altitudes:
        print(altitude)
        snowfall_study = SafranSnowfall1Day(altitude=altitude, season=season)
        temperature_study = SafranTemperature(altitude=altitude, season=season)
        precipitation_study = SafranPrecipitation1Day(altitude=altitude, season=season)
        # Compute optimal temperature
        all_time_series = [temperature_study.all_daily_series, snowfall_study.all_daily_series,
                           precipitation_study.all_daily_series]
        *_, optimal_temperature = compute_daily_snowfall_fraction(plot=False, all_time_series=all_time_series)
        altitude_to_optimal_temperature[altitude] = optimal_temperature
        # Compute temperature corresponding to maxima
        maxima_temperatures = []
        for year in snowfall_study.ordered_years[:2]:
            a = temperature_study.year_to_daily_time_serie_array[year]
            if temperature_at_maxima:
                annual_maxima_index = snowfall_study.year_to_annual_maxima_tuple_indices_for_daily_time_series[year]
                temp = [a[tuple_idx] for tuple_idx in annual_maxima_index]
            else:
                temp = a.flatten()
            maxima_temperatures.append(temp)
        altitude_to_maxima_temperature[altitude] = np.concatenate(maxima_temperatures)

    ax = plt.gca()
    # Plot the optimal temperature
    ax.plot(altitudes, [altitude_to_optimal_temperature[a] for a in altitudes], marker='o', linestyle='--', label='Optimal temperature for snowfall')

    # Plot the box plot
    width = 150
    ax.boxplot([altitude_to_maxima_temperature[a] for a in altitudes], positions=altitudes, widths=width)
    ax.set_xlim([min(altitudes) - width, max(altitudes) + width])
    suffix = 'at maxima of snowfall' if temperature_at_maxima else ''
    ylabel = 'Daily {} temperature {} ({})'.format(season_to_str(season), suffix, temperature_study.variable_class.UNIT)
    print(ylabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Altitude (m)')
    ax.legend()
    ax.grid()
    ax.set_ylim([-8, 5])

    plt.show()


if __name__ == '__main__':
    # altitudes = [900, 1200]
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
    distribution_temperature_of_maxima_of_snowfall(altitudes=altitudes,
                                                   temperature_at_maxima=False)
