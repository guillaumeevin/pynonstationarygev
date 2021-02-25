from itertools import chain
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.trend_test.visualizers import \
    StudyVisualizerForNonStationaryTrends
from extreme_fit.distribution.gev.gev_params import GevParams
from projects.exceeding_snow_loads.section_data.main_example_swe_total_plot import tuples_for_examples_paper1


def extract_time_serimes_with_worst_number_of_zeros(altitude_to_visualizer, nb_worst_examples):
    # Extract all the values
    l = []
    for a, v in altitude_to_visualizer.items():
        l.extend([(a, v, m, p) for m, p in v.massif_name_to_psnow.items()])
    # Sort them and keep the worst examples
    l = sorted(l, key=lambda t: t[-1])
    if nb_worst_examples is not None:
        l = l[:nb_worst_examples]
    print('Worst examples:')
    for a, v, m, p in l:
        print(a, m, p)
        print('Last standard quantile (depends on the number of data):', last_quantile(p))
    return l


def plot_qqplot_for_time_series_with_missing_zeros(
        altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
        nb_worst_examples=3):
    l = extract_time_serimes_with_worst_number_of_zeros(altitude_to_visualizer, nb_worst_examples)
    for a, v, m, p in l:
        v.qqplot(m)


def plot_intensity_against_gumbel_quantile_for_time_series_with_missing_zeros(
        altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
        nb_worst_examples=3):
    l = extract_time_serimes_with_worst_number_of_zeros(altitude_to_visualizer, nb_worst_examples)
    for a, v, m, p in l:
        v.intensity_plot(m, p)


def plot_intensity_against_gumbel_quantile_for_3_examples(
        altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    for color, altitude, m in tuples_for_examples_paper1():
        v = altitude_to_visualizer[altitude]
        v.intensity_plot(m, color)
        v.qqplot(m, color)


def plot_full_diagnostic(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    for v in altitude_to_visualizer.values():
        for m in v.massif_name_to_trend_test_that_minimized_aic.keys():
            v.intensity_plot(m)
            v.qqplot(m)


def plot_qqplot_for_time_series_examples(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    marker_altitude_massif_name_for_paper1 = tuples_for_examples_paper1()
    for color, a, m in marker_altitude_massif_name_for_paper1:
        v = altitude_to_visualizer[a]
        v.qqplot(m, color)


def plot_exceedance_psnow(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    altitudes = list(altitude_to_visualizer.keys())
    percentages_list = []
    for a, v in altitude_to_visualizer.items():
        percentages = v.mean_percentage_of_standards_for_massif_names_with_years_without_snow()
        percentages_list.append(percentages)

    ax2 = plt.gca()
    ax = ax2.twinx()
    ax2.bar(altitudes, [len(v.massifs_names_with_year_without_snow) for v in altitude_to_visualizer.values()],
            width=150, label='Number of time series', edgecolor='black', hatch='x', fill=False)
    ax2.set_ylabel('Number of time series with some\nannual maxima equal to zero, i.e. with $P(Y > 0) < 1$')

    mean = [p[1] for p in percentages_list]
    alpha = 0.2
    color = 'blue'
    label_name = 'ratio'

    confidence_interval_str = ' {}'.format(AbstractExtractEurocodeReturnLevel.percentage_confidence_interval)
    confidence_interval_str += '% confidence interval'
    ax.plot(altitudes, mean, linestyle='--', marker='o', color=color,
            label=label_name)
    lower_bound = [p[0] for p in percentages_list]
    upper_bound = [p[2] for p in percentages_list]

    ax.fill_between(altitudes, lower_bound, upper_bound, color=color, alpha=alpha,
                    label=label_name + confidence_interval_str)
    # Plot error bars
    yerr = np.array([[d[1] - d[0], d[2] - d[1]] for d in zip(lower_bound, mean, upper_bound)]).transpose()
    ax.bar(altitudes, mean, ecolor='black', capsize=5, yerr=yerr)

    ax.set_xticks(altitudes)
    ax.set_yticks([j * 20 for j in range(6)])
    ax2.set_yticks([j * 4 for j in range(6)])

    ax.set_xlabel('Altitude (m)')
    ax.set_ylabel('Mean ratio, i.e. French standards divided by return levels (%)')
    size = 10
    ax.legend(loc='upper right', prop={'size': size})
    ax.grid()
    ax2.legend(loc='upper left', prop={'size': size})

    plt.show()


def plot_hist_psnow(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """Plot an histogram of psnow containing data from all the visualizers given as argument"""
    """
    Altitude with psnow < 1
    300 13 ['Chablais', 'Aravis', 'Mont-Blanc', 'Bauges', 'Beaufortain', 'Chartreuse', 'Belledonne', 'Maurienne', 'Vanoise', 'Oisans', 'Devoluy', 'Haut_Var-Haut_Verdon', 'Mercantour']
    600 4 ['Parpaillon', 'Ubaye', 'Haut_Var-Haut_Verdon', 'Mercantour']
    900 1 ['Mercantour']
    1200 1 ['Mercantour']

    300m all massifs with data except Vercors (for 9 we don't have data)
    600m the 4 most southern massif
    900 and 1200 for the most southern of all massif

    """
    ax = plt.gca()
    # Gather the data
    for a, v in altitude_to_visualizer.items():
        m = v.massifs_names_with_year_without_snow
        if len(m) > 0:
            print(a, len(m), m)
    data = [list(v.massif_name_to_psnow.values()) for v in altitude_to_visualizer.values()]
    data = list(chain.from_iterable(data))
    print(sorted(data))
    data = np.array(data)
    percentage_of_one = sum([d == 1 for d in data]) / len(data)
    print(percentage_of_one)
    data = [d for d in data if d < 1]
    # Plot histogram
    nb_bins = 7
    percentage = False
    weights = [1 / len(data) for _ in data] if percentage else None
    count, *_ = ax.hist(data, bins=nb_bins, range=(0.3, 1), weights=weights, rwidth=0.8)
    # Set x ticks for the histogram
    ax.set_xticks([0.1 * i + 0.3 for i in range(nb_bins + 1)])
    ax.set_xlim([0.3, 1])
    size = 10
    if weights:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('Probability to have an annual maxima equal to zero, i.e. P(Y > 0)', fontsize=size)
    s = '%' if percentage else 'Number'
    ylabel = '{} of time series with some\nannual maxima equal to zero, i.e. with $P(Y > 0) < 1$'
    ax.set_ylabel(ylabel.format(s), fontsize=size)
    if not percentage:
        # Set y ticks for the histogram
        max_data = int(max(count))
        ticks = [i for i in range(max_data + 2)]
        ax.set_yticks(ticks)
        ax.set_ylim([min(ticks), max(ticks)])
    ax.tick_params(labelsize=size)
    plt.show()


def last_quantile(psnow):
    n = int(60 * psnow)
    last_proba = n / (n + 1)
    standard_gumbel = GevParams(0.0, 1.0, 0.0)
    return standard_gumbel.quantile(last_proba)


def non_stationarity_psnow(altitude_to_visualizer):
    all_relative_change_in_psnow = []
    for a, v in altitude_to_visualizer.items():
        all_relative_change_in_psnow.extend(list(v.massif_name_to_relative_change_in_psnow.values()))
    s = pd.Series(all_relative_change_in_psnow)
    print(s.describe())


if __name__ == '__main__':
    """
    Worst examples:
    300 Mercantour 0.38333333333333336
    3.1568494936985307
    300 Haut_Var-Haut_Verdon 0.6
    3.5972497046789322
    600 Mercantour 0.75
    3.817672071062871
    
    For the two time series with less values:
    300 Mercantour 1.0857026816954518
    300 Haut_Var-Haut_Verdon 0.8446498197950775

    """
    altitudes = [300, 600, 900, 1200, 1500, 1800][:3]
    # altitudes = ALL_ALTITUDES_WITHOUT_NAN
    # altitudes = [900, 1800, 2700]
    altitude_to_visualizer = {altitude: StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude),
                                                                              select_only_acceptable_shape_parameter=True,
                                                                              fit_method=MarginFitMethod.extremes_fevd_mle,
                                                                              multiprocessing=True,
                                                                              fit_gev_only_on_non_null_maxima=True,
                                                                              fit_only_time_series_with_ninety_percent_of_non_null_values=True,
                                                                              save_to_file=True,
                                                                              show=False)
                              for altitude in altitudes}

    # plot_qqplot_wrt_standard_gumbel(altitude_to_visualizer)
    plot_hist_psnow(altitude_to_visualizer)
    # plot_intensity_against_gumbel_quantile_for_time_series_with_missing_zeros(altitude_to_visualizer, nb_worst_examples=None)
    # plot_exceedance_psnow(altitude_to_visualizer)
    # non_stationarity_psnow(altitude_to_visualizer)

    # plot_qqplot_for_time_series_examples(altitude_to_visualizer)
    # plot_intensity_against_gumbel_quantile_for_time_series_with_missing_zeros(altitude_to_visualizer, nb_worst_examples=3)
