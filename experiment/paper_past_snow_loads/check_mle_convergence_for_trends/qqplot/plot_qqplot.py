from typing import Dict

from experiment.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from experiment.paper_past_snow_loads.data.main_example_swe_total_plot import marker_altitude_massif_name_for_paper1
from experiment.paper_past_snow_loads.paper_main_utils import load_altitude_to_visualizer
from experiment.paper_past_snow_loads.study_visualizer_for_non_stationary_trends import \
    StudyVisualizerForNonStationaryTrends
from experiment.trend_analysis.univariate_test.extreme_trend_test.abstract_gev_trend_test import AbstractGevTrendTest


def plot_qqplot_wrt_standard_gumbel(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
                                    plot_all=False):
    if plot_all:
        pass
    else:
        # Plot only some examples
        plot_qqplot_for_time_series_examples(altitude_to_visualizer)
        plot_qqplot_for_time_series_with_missing_zeros(altitude_to_visualizer)


def plot_qqplot_for_time_series_with_missing_zeros(altitude_to_visualizer: Dict[int, StudyVisualizerForNonStationaryTrends],
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


if __name__ == '__main__':
    # for the five worst, 300 is interesti
    altitudes = [300, 900, 1800, 2700]
    altitude_to_visualizer = {altitude:  StudyVisualizerForNonStationaryTrends(CrocusSnowLoadTotal(altitude=altitude),
                                                                               multiprocessing=True)
                              for altitude in altitudes}
    plot_qqplot_wrt_standard_gumbel(altitude_to_visualizer)

