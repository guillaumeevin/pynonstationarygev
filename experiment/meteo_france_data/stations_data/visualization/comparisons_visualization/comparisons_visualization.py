from itertools import chain
from typing import Dict, List

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    VisualizationParameters
from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis, MASSIF_COLUMN_NAME, \
    REANALYSE_STR, ALTITUDE_COLUMN_NAME
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from experiment.trend_analysis.univariate_test.utils import compute_gev_change_point_test_results
from extreme_estimator.extreme_models.result_from_fit import ResultFromIsmev
from extreme_estimator.extreme_models.utils import r, safe_run_r_estimator, ro
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates

DISTANCE_COLUMN_NAME = 'distance'


class ComparisonsVisualization(VisualizationParameters):

    def __init__(self, altitudes=None, keep_only_station_without_nan_values=False, margin=150,
                 normalize_observations=False):
        self.keep_only_station_without_nan_values = keep_only_station_without_nan_values
        if self.keep_only_station_without_nan_values:
            self.nb_columns = 5
        else:
            self.nb_columns = 7
        # Load altitude_to_comparison dictionary
        super().__init__()
        self.altitude_to_comparison = {}  # type: Dict[int, ComparisonAnalysis]
        for altitude in altitudes:
            comparison = ComparisonAnalysis(altitude=altitude,
                                            normalize_observations=normalize_observations,
                                            one_station_per_massif=False,
                                            transformation_class=None,
                                            margin=margin,
                                            keep_only_station_without_nan_values=keep_only_station_without_nan_values)
            self.altitude_to_comparison[altitude] = comparison

    @property
    def comparisons(self) -> List[ComparisonAnalysis]:
        return list(self.altitude_to_comparison.values())

    @property
    def nb_plot(self):
        return sum([c.nb_massifs for c in self.comparisons])

    @property
    def massifs(self):
        return sorted(set(chain(*[c.intersection_massif_names for c in self.comparisons])))

    def _visualize_main(self, plot_function, title=''):
        nb_rows = math.ceil(self.nb_plot / self.nb_columns)
        fig, axes = plt.subplots(nb_rows, self.nb_columns, figsize=self.figsize)
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
        axes = axes.flatten()

        ax_idx = 0
        for massif in self.massifs:
            for c in [c for c in self.comparisons if massif in c.intersection_massif_names]:
                self._visualize_ax_main(plot_function, c, massif, axes[ax_idx])
                ax_idx += 1
        plt.suptitle(title)
        plt.show()

    def _visualize_ax_main(self, plot_function, comparison: ComparisonAnalysis, massif, ax=None, show=False):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)
        ax2 = ax.twinx()

        df = comparison.df_merged_intersection_clean.copy()
        ind = df[MASSIF_COLUMN_NAME] == massif
        df.drop([MASSIF_COLUMN_NAME], axis=1, inplace=True)
        assert sum(ind) > 0
        df = df.loc[ind]  # type: pd.DataFrame
        colors_station = ['r', 'tab:orange', 'tab:purple', 'm', 'k']
        # Compute a distance column
        ind_location = df.index.str.contains(REANALYSE_STR)
        df[DISTANCE_COLUMN_NAME] = 0
        for coordinate_name in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            center = df.loc[ind_location, coordinate_name].values[0]
            df[DISTANCE_COLUMN_NAME] += (center - df[coordinate_name]).pow(2)
        df[DISTANCE_COLUMN_NAME] = df[DISTANCE_COLUMN_NAME].pow(0.5)
        df[DISTANCE_COLUMN_NAME] = (df[DISTANCE_COLUMN_NAME] / 1000).round(1)

        for color, (i, s) in zip(colors_station, df.iterrows()):
            label = i
            label += ' ({}m)'.format(s[ALTITUDE_COLUMN_NAME])
            label += ' ({}km)'.format(s[DISTANCE_COLUMN_NAME])
            s_values = s.iloc[3:-1].to_dict()
            years, maxima = list(s_values.keys()), list(s_values.values())

            plot_color = color if REANALYSE_STR not in label else 'g'

            plot_function(ax, ax2, years, maxima, label, plot_color)
            ax.set_title('{} at {}m'.format(massif, comparison.altitude))
            ax.legend(prop={'size': 5})

        if show:
            plt.show()

    def visualize_maximum(self):
        return self._visualize_main(self.plot_maxima, 'Recent trend of Annual maxima of snowfall')

    def plot_maxima(self, ax, ax2, years, maxima, label, plot_color):
        if self.keep_only_station_without_nan_values:
            # Run trend test to improve the label
            starting_years = years[:-1]
            trend_test_res, best_idxs = compute_gev_change_point_test_results(multiprocessing=True,
                                                                              maxima=maxima, starting_years=starting_years,
                                                                              trend_test_class=GevLocationChangePointTest,
                                                                              years=years)
            best_idx = best_idxs[0]
            most_likely_year = years[best_idx]
            most_likely_trend_type = trend_test_res[best_idx][0]
            display_trend_type = AbstractUnivariateTest.get_display_trend_type(real_trend_type=most_likely_trend_type)
            label += "\n {} starting in {}".format(display_trend_type, most_likely_year)
            # Display the nllh against the starting year
            step = 10
            ax2.plot(starting_years[::step], [t[3] for t in trend_test_res][::step], color=plot_color, marker='o')
            ax2.plot(starting_years[::step], [t[4] for t in trend_test_res][::step], color=plot_color, marker='x')
        # Plot maxima
        ax.plot(years, maxima, label=label, color=plot_color)

    def visualize_gev(self):
        return self._visualize_main(self.plot_gev)

    def plot_gev(self, ax, ax2, s_values, label, plot_color):
        # todo should I normalize here ?
        # fit gev
        data = list(s_values.values())
        res = safe_run_r_estimator(function=r('gev.fit'), xdat=ro.FloatVector(data),
                                   use_start=True)
        res = ResultFromIsmev(res, {})
        gev_params = res.stationary_gev_params

        lim = 1.5 * max(data)
        x = np.linspace(0, lim, 1000)
        y = gev_params.density(x)
        # display the gev distribution that was obtained
        ax.plot(x, y, label=label, color=plot_color)
