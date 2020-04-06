import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranRainfall1Day, SafranTemperature, \
    SafranSnowfall1Day


def plot_snowfall_fraction(ax, altitude, temperature_array, snowfall_array, rainfall_array):
    if ax is None:
        ax = plt.gca()
    lim = 10
    bins = np.linspace(-lim, lim, num=(4 * 2 * lim) + 1)
    inds = np.digitize(temperature_array, bins)
    snowfall_fractions = []
    for j in range(1, len(bins)):
        mask = inds == j
        fraction = np.mean(snowfall_array[mask]) / np.mean(snowfall_array[mask] + rainfall_array[mask])
        snowfall_fractions.append(fraction)
    new_snowfall_fractions = 100 * gaussian_filter1d(snowfall_fractions, sigma=0.5)
    x = bins[:-1] + 0.125

    ax.plot(x, new_snowfall_fractions, label=altitude)
    ax.set_ylabel('Snowfall fraction (%)')
    ax.set_xlabel('Daily surface air temperature (T)')
    ax.legend()


def daily_snowfall_fraction(ax, altitude, year_min=1959, year_max=2019):
    temperature_study = SafranTemperature(altitude=altitude)
    study_rainfall = SafranRainfall1Day(altitude=altitude)
    study_snowfall = SafranSnowfall1Day(altitude=altitude)
    all_time_series = [temperature_study.all_daily_series, study_snowfall.all_daily_series,
                       study_rainfall.all_daily_series]
    plot_snowfall_fraction(ax, *[np.concatenate(t) for t in all_time_series])


def daily_snowfall_fraction_wrt_altitude():
    ax = plt.gca()
    for altitude in [900, 1800, 2700]:
        daily_snowfall_fraction(ax, altitude)
    plt.show()


if __name__ == '__main__':
    daily_snowfall_fraction_wrt_altitude()
    # daily_snowfall_fraction(altitude=900)
