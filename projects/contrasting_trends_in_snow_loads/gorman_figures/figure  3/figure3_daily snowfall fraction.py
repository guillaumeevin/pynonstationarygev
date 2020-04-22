import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranRainfall1Day, SafranTemperature, \
    SafranSnowfall1Day

Altitudes_groups = [[300, 600, 900], [1200, 1500, 1800], [2100, 2400, 2700], [3000, 3300, 3600]]


def compute_smoother_snowfall_fraction(altitudes, year_min, year_max, lim, massif_names=None):
    all_time_series = np.concatenate([get_time_series(altitude, year_min, year_max, massif_names) for altitude in altitudes], axis=1)
    temperature_array, snowfall_array, rainfall_array = all_time_series
    # nb_bins_for_one_celsius = 4
    nb_bins_for_one_celsius = 2
    bins = np.linspace(-lim, lim, num=int((nb_bins_for_one_celsius * 2 * lim) + 1))
    inds = np.digitize(temperature_array, bins)
    snowfall_fractions = []
    for j in range(1, len(bins)):
        mask = inds == j
        fraction = np.mean(snowfall_array[mask]) / np.mean(snowfall_array[mask] + rainfall_array[mask])
        snowfall_fractions.append(fraction)
    x = bins[:-1] + 0.125
    return np.array(snowfall_fractions), x


def daily_snowfall_fraction(ax, altitudes, year_min=1959, year_max=2019, massif_names=None):
    lim = 6
    snowfall_fractions, x = compute_smoother_snowfall_fraction(altitudes, year_min, year_max, lim=lim,
                                                               massif_names=massif_names)
    sigma = 1.0
    snowfall_fractions = 100 * gaussian_filter1d(snowfall_fractions, sigma=sigma)
    # Plot results
    label = '{}-{} at {}'.format(year_min, year_max, ' & '.join(['{} m'.format(a) for a in altitudes]))
    ax.set_xlim([-lim, lim])
    ax.plot(x, snowfall_fractions, label=label, linewidth=5)
    ax.set_ylabel('Snowfall fraction (%)')
    end_plot(ax, sigma)


def snowfall_fraction_difference_between_periods(ax, altitudes, year_min=1959, year_middle=1989, year_max=2019):
    lim = 4
    snowfall_fractions_past, x = compute_smoother_snowfall_fraction(altitudes, year_min, year_middle, lim=lim)
    snowfall_fractions_recent, x = compute_smoother_snowfall_fraction(altitudes, year_middle + 1, year_max, lim=lim)
    # Ensure that we do not divide by something too small and that the analysis becomes useless
    last_percentage_of_interest = 0.5
    mask = snowfall_fractions_past > last_percentage_of_interest
    x = x[mask]
    snowfall_percentage = 100 * (snowfall_fractions_recent[mask] - snowfall_fractions_past[mask]) / snowfall_fractions_past[mask]
    sigma = 1.0
    snowfall_percentage = gaussian_filter1d(snowfall_percentage, sigma=sigma)
    label = '{}'.format(' & '.join(['{} m'.format(a) for a in altitudes]))
    ax.plot(x, snowfall_percentage, label=label, linewidth=4)
    ax.set_ylabel('Relative change of Snowfall fraction {}-{}\n'
                  'compared to Snowfall fraction {}-{}\n'
                  'until the latter reach a fraction of 0.5 (%)'.format(year_middle + 1, year_max, year_min, year_middle))
    end_plot(ax, sigma)


def end_plot(ax, sigma):
    ax.set_xlabel('Daily surface air temperature (Celsius)')
    ax.set_title('Snowfall fraction is smoothed with a gaussian filter with std={}'.format(sigma))
    ax.grid(b=True)
    ax.legend()


def get_time_series(altitude, year_min, year_max, massif_names=None):
    temperature_study = SafranTemperature(altitude=altitude, year_min=year_min, year_max=year_max)
    study_rainfall = SafranRainfall1Day(altitude=altitude, year_min=year_min, year_max=year_max)
    study_snowfall = SafranSnowfall1Day(altitude=altitude, year_min=year_min, year_max=year_max)
    studies = [temperature_study, study_rainfall, study_snowfall]
    if massif_names is None:
        all
    all_time_series = [temperature_study.all_daily_series,
                       study_snowfall.all_daily_series,
                       study_rainfall.all_daily_series]
    if massif_names is not None:
        # Select only the index corresponding to the massif of interest
        massif_ids = [temperature_study.massif_name_to_massif_id[name] for name in massif_names]
        all_time_series = [a[:, massif_ids] for a in all_time_series]
    all_time_series = [np.concatenate(t) for t in all_time_series]
    all_time_series = np.array(all_time_series)
    return all_time_series


def daily_snowfall_fraction_wrt_altitude(fast=True):
    ax = plt.gca()
    groups = [[1800]] if fast else Altitudes_groups
    groups = [[a] for a in [900, 1200, 1500]]
    massif_names = None
    massif_names = ['Vercors']
    for altitudes in groups:
        daily_snowfall_fraction(ax, altitudes, massif_names=massif_names, year_max=1999)
    plt.show()


def ratio_of_past_and_recent_daily_snowfall_fraction_wrt_altitude():
    ax = plt.gca()
    for altitudes in Altitudes_groups[:]:
        snowfall_fraction_difference_between_periods(ax, altitudes)
    plt.show()


def daily_snowfall_fraction_wrt_time():
    ax = plt.gca()
    for altitudes in Altitudes_groups:
        for year_min, year_max in [(1959, 1989), (1990, 2019)]:
            daily_snowfall_fraction(ax, altitudes, year_min, year_max)
    plt.show()


if __name__ == '__main__':
    daily_snowfall_fraction_wrt_altitude(fast=False)
    # daily_snowfall_fraction_wrt_time()
    # ratio_of_past_and_recent_daily_snowfall_fraction_wrt_altitude()
