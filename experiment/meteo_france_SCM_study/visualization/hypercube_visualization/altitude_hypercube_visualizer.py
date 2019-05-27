import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_test.abstract_univariate_test import AbstractUnivariateTest
from utils import get_display_name_from_object_type


class AltitudeHypercubeVisualizer(AbstractHypercubeVisualizer):

    @property
    def altitudes(self):
        return self.tuple_values(idx=0)

    @property
    def display_trend_type_to_style(self):
        return self.trend_test_class.display_trend_type_to_style()

    @property
    def display_trend_types(self):
        return self.display_trend_type_to_style.keys()

    @property
    def nb_axes(self):
        return 2

    def trend_type_to_series(self, reduction_function):
        # Map each trend type to its serie with percentages
        return {display_trend_type: self.trend_type_reduction(reduction_function, display_trend_type)[0]
                for display_trend_type in self.display_trend_types}

    def trend_type_reduction(self, reduction_function, display_trend_type):
        # Reduce df_bool df to a serie s_trend_type_percentage
        df_bool = self.df_hypercube_trend_type.isin(AbstractUnivariateTest.get_real_trend_types(display_trend_type))
        s_trend_type_percentage = reduction_function(df_bool)
        assert isinstance(s_trend_type_percentage, pd.Series)
        assert not isinstance(s_trend_type_percentage.index, pd.MultiIndex)
        s_trend_type_percentage *= 100
        # Reduce df_strength to a serie s_trend_strength
        df_strength = self.df_hypercube_trend_strength[df_bool]
        s_trend_strength = reduction_function(df_strength)
        # Group result
        series = [s_trend_type_percentage, s_trend_strength]
        return series, df_bool

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False, subtitle=None):
        def reduction_function_with_level(df_bool):
            return reduction_function(df_bool) if level is None else reduction_function(df_bool, level)

        if subtitle is None:
            subtitle = self.study.variable_name
        return {subtitle: reduction_function_with_level}

    def get_title_plot(self, xlabel, ax_idx=None):
        labels = ['altitudes', 'starting years', 'massifs']
        assert xlabel in labels, xlabel
        if ax_idx == 1:
            return '% of change per year for the parameter value'
        elif ax_idx == 0:
            return '% of trend type'
        else:
            labels.remove(xlabel)
            if xlabel != 'starting years':
                labels.remove('starting years')
            common_txt = 'averaged on {}'.format(' & '.join(labels))
            return common_txt

    def visualize_trend_test_evolution(self, reduction_function, xlabel, xlabel_values, axes=None, marker='o',
                                       subtitle=''):
        if axes is None:
            fig, axes = plt.subplots(self.nb_axes, 1, figsize=self.study_visualizer.figsize)

        trend_type_to_series = self.trend_type_to_series(reduction_function)
        for ax_idx, ax in enumerate(axes):
            for trend_type in self.display_trend_types:
                style = self.display_trend_type_to_style[trend_type]
                percentages_values = trend_type_to_series[trend_type][ax_idx]
                ax.plot(xlabel_values, percentages_values, style + marker, label=trend_type)

            if ax_idx == 0:
                # Global information
                ax.set_ylabel(self.get_title_plot(xlabel, ax_idx=0))
                ax.set_yticks(list(range(0, 101, 10)))
            else:
                ax.set_ylabel(self.get_title_plot(xlabel, ax_idx=ax_idx))

            # Common function functions
            ax.set_xlabel(xlabel)
            ax.grid()
            ax.set_xticks(xlabel_values)
            ax.legend()

        title = 'Evolution of {} trends (significative or not) wrt to the {} with {}'.format(subtitle, xlabel,
                                                                                             self.trend_test_name)
        title += '\n ' + self.get_title_plot(xlabel)
        plt.suptitle(title)
        self.show_or_save_to_file(specific_title=title)

    def visualize_trend_test_repartition(self, reduction_function, axes=None, subtitle=''):
        if axes is None:
            nb_trend_type = len(self.display_trend_type_to_style)
            fig, axes = plt.subplots(self.nb_axes, nb_trend_type, figsize=self.study_visualizer.figsize)

        for i, axes_row in enumerate(axes):
            trend_type_to_serie = {k: v[i] for k, v in self.trend_type_to_series(reduction_function).items()}
            vmax = max([s.max() for s in trend_type_to_serie.values()])
            vmin = min([s.min() for s in trend_type_to_serie.values()])
            vmax = max(vmax, 0.01)
            for ax, trend_type in zip(axes_row, self.display_trend_types):
                s_percentages = trend_type_to_serie[trend_type]
                massif_to_value = dict(s_percentages)
                cmap = self.trend_test_class.get_cmap_from_trend_type(trend_type)
                self.study.visualize_study(ax, massif_to_value, show=False, cmap=cmap, label=None, vmax=vmax, vmin=vmin)
                ax.set_title(trend_type)
            row_title = self.get_title_plot(xlabel='massifs', ax_idx=i)
            StudyVisualizer.clean_axes_write_title_on_the_left(axes_row, row_title, left_border=None)

        # Global information
        title = 'Repartition of {} trends (significative or not) with {}'.format(subtitle, self.trend_test_name)
        title += '\n ' + self.get_title_plot('massifs')
        plt.suptitle(title)
        self.show_or_save_to_file(specific_title=title)

    @property
    def altitude_index_level(self):
        return 0

    @property
    def massif_index_level(self):
        return 1

    def visualize_year_trend_test(self, axes=None, marker='o', add_detailed_plots=False):
        def year_reduction(df):
            # Take the mean with respect to all the first axis indices
            return df.mean(axis=0)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(year_reduction,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='starting years',
                                                xlabel_values=self.starting_years, axes=axes, marker=marker,
                                                subtitle=subtitle)

    @staticmethod
    def index_reduction(df, level):
        # Take the sum with respect to the years, replace any missing data with np.nan
        df = df.any(axis=1)
        # Take the mean with respect to the level of interest
        return df.mean(level=level)

    def visualize_altitude_trend_test(self, axes=None, marker='o', add_detailed_plots=False):
        for subtitle, reduction_function in self.subtitle_to_reduction_function(self.index_reduction,
                                                                                level=self.altitude_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='altitudes',
                                                xlabel_values=self.altitudes, axes=axes, marker=marker,
                                                subtitle=subtitle)

    def visualize_massif_trend_test(self, axes=None, add_detailed_plots=False):
        for subtitle, reduction_function in self.subtitle_to_reduction_function(self.index_reduction,
                                                                                level=self.massif_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_repartition(reduction_function, axes, subtitle=subtitle)


class Altitude_Hypercube_Year_Visualizer(AltitudeHypercubeVisualizer):

    def get_title_plot(self, xlabel, ax_idx=None):
        if ax_idx == 2:
            return 'mean starting year'
        return super().get_title_plot(xlabel, ax_idx)

    @property
    def nb_axes(self):
        return 3

    @staticmethod
    def index_reduction(df, level):
        # Take the sum with respect to the years, replace any missing data with np.nan
        df = df.sum(axis=1).replace(0.0, np.nan)
        # Take the mean with respect to the level of interest
        return df.mean(level=level)

    def trend_type_reduction(self, reduction_function, display_trend_type):
        series, df_bool = super().trend_type_reduction(reduction_function, display_trend_type)
        # Create df argmax
        df = df_bool.copy()
        df = (df * df.columns)[df_bool]
        # Reduce and append
        serie = reduction_function(df)
        series.append(serie)
        return series, df_bool

