from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranTemperature
import numpy as np
import matplotlib.pyplot as plt


def distribution_temperature_of_maxima_of_snowfall(altitudes):
    # Load the temperature corresponding to the maxima of snowfall
    altitude_to_maxima_temperature = {}
    for altitude in altitudes:
        print(altitude)
        snowfall_study = SafranSnowfall1Day(altitude=altitude)
        temperature_study = SafranTemperature(altitude=altitude)
        # temperature_study = SafranTemperature(altitude=altitude)
        maxima_temperatures = []
        for year in snowfall_study.ordered_years[:2]:
            annual_maxima_index = snowfall_study.year_to_annual_maxima_tuple_indices_for_daily_time_series[year]
            a = temperature_study.year_to_daily_time_serie_array[year]
            temp = [a[tuple_idx] for tuple_idx in annual_maxima_index]
            maxima_temperatures.append(temp)
        altitude_to_maxima_temperature[altitude] = np.concatenate(maxima_temperatures)

    # Plot the box plot
    ax = plt.gca()
    width = 150
    ax.boxplot(altitude_to_maxima_temperature.values(), positions=altitudes, widths=width)
    ax.set_xlim([min(altitudes) - width, max(altitudes) + width])
    ax.set_ylabel('Temperature for maxima of snowfall ({})'.format(temperature_study.variable_class.UNIT))
    ax.set_xlabel('Altitude (m)')
    ax.grid()

    plt.show()


if __name__ == '__main__':
    distribution_temperature_of_maxima_of_snowfall(altitudes=[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000])
