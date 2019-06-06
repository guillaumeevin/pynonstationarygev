from collections import OrderedDict
from itertools import chain
from typing import Dict, List

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    VisualizationParameters
from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis, MASSIF_COLUMN_NAME, \
    REANALYSE_STR, ALTITUDE_COLUMN_NAME, STATION_COLUMN_NAME
from experiment.trend_analysis.univariate_test.abstract_gev_change_point_test import GevLocationChangePointTest
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from experiment.trend_analysis.univariate_test.utils import compute_gev_change_point_test_results
from extreme_estimator.extreme_models.result_from_fit import ResultFromIsmev
from extreme_estimator.extreme_models.utils import r, safe_run_r_estimator, ro
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates

DISTANCE_COLUMN_NAME = 'distance'
path_df_location_to_value_csv_example = r'/home/erwan/Documents/projects/spatiotemporalextremes/experiment/meteo_france_data/stations_data/csv/example.csv'


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

    def _visualize_main(self, plot_function, title='', show=True):
        nb_rows = math.ceil(self.nb_plot / self.nb_columns)
        fig, axes = plt.subplots(nb_rows, self.nb_columns, figsize=self.figsize)
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
        axes = axes.flatten()

        ax_idx = 0
        tuple_location_to_values = {}
        index = None
        for massif in self.massifs:
            for c in [c for c in self.comparisons if massif in c.intersection_massif_names]:
                res = self._visualize_ax_main(plot_function, c, massif, axes[ax_idx])
                ax_idx += 1
                for station_name, ordered_dict in res:
                    if index is None:
                        index = list(ordered_dict.keys())
                    else:
                        assert all([i == k for i, k in zip(index, ordered_dict.keys())])
                    tuple_location_to_values[(c.altitude, massif, station_name)] = list(ordered_dict.values())

        plt.suptitle(title)
        if show:
            plt.show()

        # Build dataframe from the dictionary
        df = pd.DataFrame(tuple_location_to_values, index=index).transpose()
        df.index.names = [ALTITUDE_COLUMN_NAME, MASSIF_COLUMN_NAME, STATION_COLUMN_NAME]
        return df

    @classmethod
    def visualize_metric(cls, df_location_to_value=None):
        # Load or update df value from example file
        if df_location_to_value is None:
            df_location_to_value = pd.read_csv(path_df_location_to_value_csv_example, index_col=[0, 1, 2])
        else:
            df_location_to_value.to_csv(path_df_location_to_value_csv_example)

        print(df_location_to_value)
        print(df_location_to_value.index)
        print(df_location_to_value.columns)

        # Display some score spatially

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
        ind_reanalysis_data = df.index.str.contains(REANALYSE_STR)
        df[DISTANCE_COLUMN_NAME] = 0
        for coordinate_name in [AbstractCoordinates.COORDINATE_X, AbstractCoordinates.COORDINATE_Y]:
            center = df.loc[ind_reanalysis_data, coordinate_name].values[0]
            df[DISTANCE_COLUMN_NAME] += (center - df[coordinate_name]).pow(2)
        df[DISTANCE_COLUMN_NAME] = df[DISTANCE_COLUMN_NAME].pow(0.5)
        df[DISTANCE_COLUMN_NAME] = (df[DISTANCE_COLUMN_NAME] / 1000).round(1)
        # Compute
        maxima_center, _ = self.get_maxima_and_year(df.loc[ind_reanalysis_data].iloc[0])

        res = []
        plot_station_ordered_value_dict = None
        for color, (i, s) in zip(colors_station, df.iterrows()):
            ordered_value_dict = OrderedDict()

            label = i
            label += ' ({}m)'.format(s[ALTITUDE_COLUMN_NAME])
            label += ' ({}km)'.format(s[DISTANCE_COLUMN_NAME])
            maxima, years = self.get_maxima_and_year(s)

            # Compute the distance between maxima and maxima_center
            # In percent the number of times the observations is stricty higher than the reanalysis
            mask = ~np.isnan(maxima_center) & ~np.isnan(maxima)
            mean_absolute_difference = np.round(np.mean(np.abs(maxima[mask] - maxima_center[mask])), 0)
            label += 'Mean absolute diff {}mm'.format(mean_absolute_difference)
            ordered_value_dict['mean absolute difference'] = mean_absolute_difference

            # metric = np.mean(np.sign(maxima[mask] - maxima_center[mask]) == 1)
            # metric = np.round(metric * 100, 0)
            # label += '{}% strictly above'.format(metric)

            plot_color = color if REANALYSE_STR not in label else 'g'
            plot_ordered_value_dict = plot_function(ax, ax2, years, maxima, label, plot_color)

            if isinstance(plot_ordered_value_dict, dict):
                if REANALYSE_STR in i:
                    plot_station_ordered_value_dict = OrderedDict([(k + ' ' + REANALYSE_STR, v)
                                                                   for k, v in plot_ordered_value_dict.items()])
                else:
                    ordered_value_dict.update(plot_ordered_value_dict)

            ax.set_title('{} at {}m'.format(massif, comparison.altitude))
            ax.legend(prop={'size': 5})

            # Store only results for the stations
            if REANALYSE_STR not in i:
                res.append((i, ordered_value_dict))

        # Add the station ordered dict
        for _, ordered_dict in res:
            ordered_dict.update(plot_station_ordered_value_dict)

        if show:
            plt.show()

        return res

    def get_maxima_and_year(self, s):
        assert isinstance(s, pd.Series)
        s_values = s.iloc[3:-1].to_dict()
        years, maxima = np.array(list(s_values.keys())), np.array(list(s_values.values()))
        return maxima, years

    def visualize_maximum(self, visualize_metric_only=None):
        show = visualize_metric_only is None or not visualize_metric_only
        df_location_to_value = self._visualize_main(plot_function=self.plot_maxima,
                                                    title='Recent trend of Annual maxima of snowfall',
                                                    show=show)
        if visualize_metric_only is not None:
            self.visualize_metric(df_location_to_value)

    def plot_maxima(self, ax, ax2, years, maxima, label, plot_color):
        ordered_dict = OrderedDict()
        if self.keep_only_station_without_nan_values:
            # Run trend test to improve the label
            starting_years = years[:-4]
            trend_test_res, best_idxs = compute_gev_change_point_test_results(multiprocessing=True,
                                                                              maxima=maxima,
                                                                              starting_years=starting_years,
                                                                              trend_test_class=GevLocationChangePointTest,
                                                                              years=years)
            best_idx = best_idxs[0]
            most_likely_year = years[best_idx]
            most_likely_trend_type = trend_test_res[best_idx][0]
            display_trend_type = AbstractUnivariateTest.get_display_trend_type(real_trend_type=most_likely_trend_type)
            label += "\n {} starting in {}".format(display_trend_type, most_likely_year)
            ordered_dict['display trend type'] = display_trend_type
            ordered_dict['most likely year'] = most_likely_year
            # Display the nllh against the starting year
            step = 1
            ax2.plot(starting_years[::step], [t[3] for t in trend_test_res][::step], color=plot_color, marker='o')
            ax2.plot(starting_years[::step], [t[4] for t in trend_test_res][::step], color=plot_color, marker='x')
        # Plot maxima
        ax.grid()
        # print("here")
        ax.plot(years, maxima, label=label, color=plot_color)
        return ordered_dict

    def visualize_gev(self):
        return self._visualize_main(self.plot_gev)

    def plot_gev(self, ax, ax2, years, maxima, label, plot_color):
        # todo should I normalize here ?
        # fit gev
        data = maxima
        res = safe_run_r_estimator(function=r('gev.fit'), xdat=ro.FloatVector(data),
                                   use_start=True)
        res = ResultFromIsmev(res, {})
        gev_params = res.stationary_gev_params

        lim = 1.5 * max(data)
        x = np.linspace(0, lim, 1000)
        y = gev_params.density(x)
        # display the gev distribution that was obtained
        ax.plot(x, y, label=label, color=plot_color)
