import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_SCM_study.visualization.hypercube_visualization.abstract_hypercube_visualizer import \
    AbstractHypercubeVisualizer
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import AbstractTrendTest
from utils import get_display_name_from_object_type


class AltitudeHypercubeVisualizer(AbstractHypercubeVisualizer):

    @property
    def altitudes(self):
        return self.tuple_values(idx=0)

    @property
    def trend_type_to_style(self):
        return self.trend_test_class.trend_type_to_style()

    @property
    def trend_types(self):
        return self.trend_type_to_style.keys()

    def trend_type_to_series(self, reduction_function):
        # Map each trend type to its serie with percentages
        trend_type_to_series = {}
        for trend_type in self.trend_types:
            # Reduce df_bool df to a serie s_trend_type_percentage
            df_bool = self.df_hypercube_trend_type.isin(AbstractTrendTest.get_trend_types(trend_type))
            s_trend_type_percentage = reduction_function(df_bool)
            assert isinstance(s_trend_type_percentage, pd.Series)
            assert not isinstance(s_trend_type_percentage.index, pd.MultiIndex)
            s_trend_type_percentage *= 100
            # Reduce df_strength to a serie s_trend_strength
            df_strength = self.df_hypercube_trend_strength[df_bool]
            s_trend_strength = reduction_function(df_strength)
            # Store results
            trend_type_to_series[trend_type] = (s_trend_type_percentage, s_trend_strength)
        return trend_type_to_series

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False, subtitle=None):
        def reduction_function_with_level(df_bool):
            return reduction_function(df_bool) if level is None else reduction_function(df_bool, level)
        if subtitle is None:
            subtitle = self.study.variable_name
        return {subtitle: reduction_function_with_level}

    def visualize_trend_test_evolution(self, reduction_function, xlabel, xlabel_values, axes=None, marker='o',
                                       subtitle=''):
        if axes is None:
            fig, axes = plt.subplots(2, 1, figsize=self.study_visualizer.figsize)

        trend_type_to_series = self.trend_type_to_series(reduction_function)
        for i, ax in enumerate(axes):
            for trend_type in self.trend_types:
                style = self.trend_type_to_style[trend_type]
                percentages_values = trend_type_to_series[trend_type][i]
                ax.plot(xlabel_values, percentages_values, style + marker, label=trend_type)

            if i == 0:
                # Plot the total value of significative values
                significative_values = trend_type_to_series[AbstractTrendTest.SIGNIFICATIVE_NEGATIVE_TREND][i] \
                                       + trend_type_to_series[AbstractTrendTest.SIGNIFICATIVE_POSITIVE_TREND][i]
                ax.plot(xlabel_values, significative_values, 'y-' + marker, label=AbstractTrendTest.SIGNIFICATIVE + ' trends')

                # Global information
                added_str = 'weighted '
                ylabel = '% averaged on massifs & {}averaged on {}'.format(added_str, xlabel)
                ylabel += ' (with uniform weights)'
                ax.set_ylabel(ylabel)

                ax.set_yticks(list(range(0, 101, 10)))

            # Common function functions
            ax.set_xlabel(xlabel)
            ax.grid()
            ax.set_xticks(xlabel_values)
            ax.legend()

        name = get_display_name_from_object_type(self.trend_test_class)
        title = 'Evolution of {} trends (significative or not) wrt to the {} with {}'.format(subtitle, xlabel,
                                                                                             name)
        plt.suptitle(title)
        self.show_or_save_to_file(specific_title=title)

    def visualize_trend_test_repartition(self, reduction_function, axes=None, subtitle=''):
        if axes is None:
            nb_trend_type = len(self.trend_test_class.trend_type_to_style())
            fig, axes = plt.subplots(1, nb_trend_type, figsize=self.study_visualizer.figsize)

        # Plot weighted percentages over the years
        # todo: implement strength plot spatially
        trend_type_to_s_percentages = {k: v[0] for k, v in self.trend_type_to_series(reduction_function).items()}
        for ax, trend_type in zip(axes, self.trend_types):
            s_percentages = trend_type_to_s_percentages[trend_type]
            massif_to_value = dict(s_percentages)
            cmap = self.trend_test_class.get_cmap_from_trend_type(trend_type)
            self.study.visualize_study(ax, massif_to_value, show=False, cmap=cmap, label=None)
            ax.set_title(trend_type)

        # Global information
        name = get_display_name_from_object_type(self.trend_test_class)
        title = 'Repartition of trends (significative or not) with {}'.format(name)
        title += '\n(in % averaged on altitudes & averaged on starting years)'
        title += 'with {} data'.format(subtitle)
        StudyVisualizer.clean_axes_write_title_on_the_left(axes, title, left_border=None)
        plt.suptitle(title)
        self.show_or_save_to_file(specific_title=title)

    @property
    def altitude_index_level(self):
        return 0

    @property
    def massif_index_level(self):
        return 1

    def visualize_year_trend_test(self, axes=None, marker='o', add_detailed_plots=False):
        def year_reduction(df_bool):
            # Take the mean with respect to all the first axis indices
            return df_bool.mean(axis=0)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(year_reduction,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='starting years',
                                                xlabel_values=self.starting_years, axes=axes, marker=marker,
                                                subtitle=subtitle)

    def visualize_altitude_trend_test(self, axes=None, marker='o', add_detailed_plots=False):
        def altitude_reduction(df_bool, level):
            # Take the mean with respect to the years
            df_bool = df_bool.mean(axis=1)
            # Take the mean with respect the massifs
            return df_bool.mean(level=level)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(altitude_reduction,
                                                                                level=self.altitude_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='altitude',
                                                xlabel_values=self.altitudes, axes=axes, marker=marker,
                                                subtitle=subtitle)

    def visualize_massif_trend_test(self, axes=None, add_detailed_plots=False):
        def massif_reduction(df_bool, level):
            # Take the mean with respect to the years
            df_bool = df_bool.mean(axis=1)
            # Take the mean with respect the altitude
            return df_bool.mean(level=level)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(massif_reduction,
                                                                                level=self.massif_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_repartition(reduction_function, axes, subtitle=subtitle)
