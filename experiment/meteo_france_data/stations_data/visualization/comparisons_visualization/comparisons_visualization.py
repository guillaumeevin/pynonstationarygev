from typing import Dict, List

import math
import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    VisualizationParameters
from experiment.meteo_france_data.stations_data.comparison_analysis import ComparisonAnalysis, MASSIF_COLUMN_NAME, \
    REANALYSE_STR, ALTITUDE_COLUMN_NAME
from itertools import chain


class ComparisonsVisualization(VisualizationParameters):

    def __init__(self, altitudes=None, keep_only_station_without_nan_values=False, margin=150):
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
                                            normalize_observations=False,
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

    def _visualize_main(self, visualization_ax_function, title=''):
        nb_rows = math.ceil(self.nb_plot / self.nb_columns)
        fig, axes = plt.subplots(nb_rows, self.nb_columns, figsize=self.figsize)
        fig.subplots_adjust(hspace=self.subplot_space, wspace=self.subplot_space)
        axes = axes.flatten()

        ax_idx = 0
        for massif in self.massifs:
            for c in [c for c in self.comparisons if massif in c.intersection_massif_names]:
                visualization_ax_function(c, massif, axes[ax_idx])
                ax_idx += 1
        plt.suptitle(title)
        plt.show()

    def visualize_maximum(self):
        return self._visualize_main(self._visualize_maximum, 'Recent trend of Annual maxima of snowfall')

    def _visualize_maximum(self, comparison: ComparisonAnalysis, massif, ax=None, show=False):
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=self.figsize)

        df = comparison.load_main_df_merged_intersection_clean()
        ind = df[MASSIF_COLUMN_NAME] == massif
        df.drop([MASSIF_COLUMN_NAME], axis=1, inplace=True)
        assert sum(ind) > 0
        df = df.loc[ind] # type: pd.DataFrame
        colors_station = ['r', 'tab:orange', 'tab:purple', 'm', 'k']
        for color, (i, s) in zip(colors_station, df.iterrows()):
            label = i
            label += ' ({}m)'.format(s[ALTITUDE_COLUMN_NAME])
            s_values = s.iloc[3:].to_dict()
            plot_color = color if REANALYSE_STR not in label else 'g'
            ax.plot(list(s_values.keys()), list(s_values.values()), label=label, color=plot_color)
            ax.legend(prop={'size': 5})
            ax.set_title('{} at {}'.format(massif, comparison.altitude))

        if show:

            plt.show()

    def visualize_gev(self):
        return self._visualize_main(self._visualize_gev)

    def _visualize_gev(self):
        pass
