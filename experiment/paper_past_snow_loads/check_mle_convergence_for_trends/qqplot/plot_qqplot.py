from itertools import chain
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.main_study_visualizer import \
    ALL_ALTITUDES_WITHOUT_NAN
from experiment.paper_past_snow_loads.data.main_example_swe_total_plot import marker_altitude_massif_name_for_paper1
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends


def plot_qqplot_for_time_series_with_missing_zeros(
        altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
        nb_worst_examples=3):
    # Extract all the values
    l = []
    for a, v in altitude_to_visualizer.items():
        l.extend([(a, v, m, p) for m, p in v.massif_name_to_psnow.items()])
    # Sort them and keep the worst examples
    l = sorted(l, key=lambda t: t[-1])[:nb_worst_examples]
    print('Worst examples:')
    for a, v, m, p in l:
        print(a, m, p)
        v.qqplot(m)


def plot_qqplot_for_time_series_examples(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    for color, a, m in marker_altitude_massif_name_for_paper1:
        v = altitude_to_visualizer[a]
        v.qqplot(m, color)


def plot_hist_psnow(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends]):
    """Plot an histogram of psnow containing data from all the visualizers given as argument"""
    # Gather the data
    data = [list(v.massif_name_to_psnow.values()) for v in altitude_to_visualizer.values()]
    data = list(chain.from_iterable(data))
    print(sorted(data))
    data = np.array(data)
    percentage_of_one = sum([d == 1 for d in data]) / len(data)
    print(percentage_of_one)
    data = [d for d in data if d < 1]
    # Plot histogram
    nb_bins = 13
    percentage = False
    weights = [1 / len(data) for _ in data] if percentage else None
    plt.hist(data, bins=nb_bins, range=(0.35, 1), weights=weights)
    plt.xticks([0.05 * i + 0.35 for i in range(nb_bins + 1)])
    if weights:
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.xlabel('Distribution of P(Y > 0) when $\\neq 1$')
    s = '%' if percentage else 'Number'
    plt.ylabel('{} of time series'.format(s))
    plt.show()


if __name__ == '__main__':
    # altitudes = [300, 600, 900, 1200, 1500, 1800][:2]
    altitudes = ALL_ALTITUDES_WITHOUT_NAN
    altitude_to_visualizer = {altitude: StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude),
                                                                              multiprocessing=True)
                              for altitude in altitudes}
    # plot_qqplot_wrt_standard_gumbel(altitude_to_visualizer)
    # plot_hist_psnow(altitude_to_visualizer)
    plot_qqplot_for_time_series_examples(altitude_to_visualizer)
    # plot_qqplot_for_time_series_with_missing_zeros(altitude_to_visualizer, nb_worst_examples=3)
