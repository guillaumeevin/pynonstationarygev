import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranRainfall1Day, SafranTemperature, \
    SafranSnowfall1Day

Altitudes_groups = [[300, 600, 900], [1200, 1500, 1800], [2100, 2400, 2700], [3000, 3300, 3600]]


def compute_smoother_snowfall_fraction(altitudes, year_min, year_max):
    all_time_series = np.concatenate([get_time_series(altitude, year_min, year_max) for altitude in altitudes], axis=1)
    temperature_array, snowfall_array, rainfall_array = all_time_series
    paper_setting = False
    if paper_setting:
        lim = 10
        sigma = 0.5
    else:
        lim = 7.5
        sigma = 1
    bins = np.linspace(-lim, lim, num=(4 * 2 * lim) + 1)
    inds = np.digitize(temperature_array, bins)
    snowfall_fractions = []
    for j in range(1, len(bins)):
        mask = inds == j
        fraction = np.mean(snowfall_array[mask]) / np.mean(snowfall_array[mask] + rainfall_array[mask])
        snowfall_fractions.append(fraction)
    new_snowfall_fractions = 100 * gaussian_filter1d(snowfall_fractions, sigma=sigma)
    x = bins[:-1] + 0.125
    return new_snowfall_fractions, sigma, x


def daily_snowfall_fraction(ax, altitudes, year_min=1959, year_max=2019):
    snowfall_fractions, sigma, x = compute_smoother_snowfall_fraction(altitudes, year_min, year_max)
    # Plot results
    label = '{}-{} at {}'.format(year_min, year_max, ' & '.join(['{} m'.format(a) for a in altitudes]))
    ax.plot(x, snowfall_fractions, label=label, linewidth=4)
    ax.set_ylabel('Snowfall fraction (%)')
    ax.set_xlabel('Daily surface air temperature (Celsius)')
    ax.set_title('Snowfall fraction is smoothed with a gaussian filter with standard deviation={}'.format(sigma))
    ax.grid(b=True)
    ax.legend()


def snowfall_fraction_difference_between_periods(ax, altitudes, year_min=1959, year_middle=1989, year_max=2019):
    snowfall_fractions_past, sigma, x = compute_smoother_snowfall_fraction(altitudes, year_min, year_middle)
    snowfall_fractions_recent, sigma, x = compute_smoother_snowfall_fraction(altitudes, year_middle + 1, year_max)
    mask = snowfall_fractions_past == 0
    x = x[mask]
    snowfall_ratio = snowfall_fractions_recent[mask] / snowfall_fractions_past[mask]
    label = '{}'.format(' & '.join(['{} m'.format(a) for a in altitudes]))
    ax.plot(x, snowfall_ratio, label=label, linewidth=4)
    ax.set_ylabel('Ratio of Snowfall fraction {}-{} divided by Snowfall fraction {}-{}'.format(year_min, year_middle,
                                                                                               year_middle + 1,
                                                                                               year_max))
    ax.set_xlabel('Daily surface air temperature (Celsius)')
    ax.set_title('Snowfall fraction is smoothed with a gaussian filter with standard deviation={}'.format(sigma))
    ax.grid(b=True)
    ax.legend()


def get_time_series(altitude, year_min, year_max):
    temperature_study = SafranTemperature(altitude=altitude, year_min=year_min, year_max=year_max)
    study_rainfall = SafranRainfall1Day(altitude=altitude, year_min=year_min, year_max=year_max)
    study_snowfall = SafranSnowfall1Day(altitude=altitude, year_min=year_min, year_max=year_max)
    all_time_series = [temperature_study.all_daily_series, study_snowfall.all_daily_series,
                       study_rainfall.all_daily_series]
    all_time_series = [np.concatenate(t) for t in all_time_series]
    all_time_series = np.array(all_time_series)
    return all_time_series


def daily_snowfall_fraction_wrt_altitude(fast=True):
    ax = plt.gca()
    groups = [[1800]] if fast else Altitudes_groups
    for altitudes in groups:
        daily_snowfall_fraction(ax, altitudes)
    plt.show()


def ratio_of_past_and_recent_daily_snowfall_fraction_wrt_altitude():
    ax = plt.gca()
    for altitudes in Altitudes_groups:
        snowfall_fraction_difference_between_periods(ax, altitudes)
    plt.show()


def daily_snowfall_fraction_wrt_time():
    ax = plt.gca()
    for altitudes in Altitudes_groups:
        for year_min, year_max in [(1959, 1989), (1990, 2019)]:
            daily_snowfall_fraction(ax, altitudes, year_min, year_max)
    plt.show()


if __name__ == '__main__':
    # daily_snowfall_fraction_wrt_altitude(fast=True)
    # daily_snowfall_fraction_wrt_time()
    ratio_of_past_and_recent_daily_snowfall_fraction_wrt_altitude()
