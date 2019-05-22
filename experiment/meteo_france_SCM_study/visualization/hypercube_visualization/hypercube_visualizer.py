import os
import os.path as op
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.trend_analysis.univariate_trend_test.abstract_trend_test import AbstractTrendTest
from utils import cached_property, VERSION_TIME, get_display_name_from_object_type


class HypercubeVisualizer(object):
    """
    A study visualizer contain some massifs and years. This forms the base DataFrame of the hypercube
    Additional index will come from the tuple.
    Tuple could contain altitudes, type of snow quantity
    """

    def __init__(self, tuple_to_study_visualizer: Dict[Tuple, StudyVisualizer],
                 trend_test_class,
                 fast=False,
                 save_to_file=False):
        self.nb_data_for_fast_mode = 7 if fast else None
        self.save_to_file = save_to_file
        self.trend_test_class = trend_test_class
        self.tuple_to_study_visualizer = tuple_to_study_visualizer  # type: Dict[Tuple, StudyVisualizer]

    # Main attributes defining the hypercube

    @property
    def trend_test_name(self):
        return get_display_name_from_object_type(self.trend_test_class)

    @cached_property
    def starting_years(self):
        starting_years = self.study_visualizer.starting_years
        if self.nb_data_for_fast_mode is not None:
            starting_years = starting_years[:self.nb_data_for_fast_mode]
        return starting_years

    @cached_property
    def tuple_to_df_trend_type(self):
        df_spatio_temporal_trend_types = [
            study_visualizer.df_trend_spatio_temporal(self.trend_test_class, self.starting_years,
                                                      self.nb_data_for_fast_mode)
            for study_visualizer in self.tuple_to_study_visualizer.values()]
        return dict(zip(self.tuple_to_study_visualizer.keys(), df_spatio_temporal_trend_types))

    def tuple_values(self, idx):
        return sorted(set([t[idx] if isinstance(t, tuple) else t for t in self.tuple_to_study_visualizer.keys()]))

    @cached_property
    def df_hypercube(self) -> pd.DataFrame:
        keys = list(self.tuple_to_df_trend_type.keys())
        values = list(self.tuple_to_df_trend_type.values())
        df = pd.concat(values, keys=keys, axis=0)
        return df

    # Some properties

    @property
    def study_title(self):
        return self.study.title

    def show_or_save_to_file(self, specific_title=''):
        if self.save_to_file:
            main_title, *_ = '_'.join(self.study_title.split()).split('/')
            filename = "{}/{}/".format(VERSION_TIME, main_title)
            filename += specific_title
            filepath = op.join(self.study.result_full_path, filename + '.png')
            dirname = op.dirname(filepath)
            if not op.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            plt.savefig(filepath)
        else:
            plt.show()

    @property
    def study_visualizer(self) -> StudyVisualizer:
        return list(self.tuple_to_study_visualizer.values())[0]

    @property
    def study(self):
        return self.study_visualizer.study

    @property
    def starting_year_to_weights(self):
        # Load uniform weights by default
        uniform_weight = 1 / len(self.starting_years)
        return {year: uniform_weight for year in self.starting_years}


class AltitudeHypercubeVisualizer(HypercubeVisualizer):

    @property
    def altitudes(self):
        return self.tuple_values(idx=0)

    @property
    def trend_type_to_style(self):
        return self.trend_test_class.trend_type_to_style()

    @property
    def trend_types(self):
        return self.trend_type_to_style.keys()

    def trend_type_to_s_percentages(self, reduction_function):
        # Map each trend type to its serie with percentages
        trend_type_to_s_percentages = {}
        for trend_type in self.trend_types:
            df_bool = (self.df_hypercube == trend_type)
            # Reduce the entire dataframe to a serie
            s_percentages = reduction_function(df_bool)
            assert isinstance(s_percentages, pd.Series)
            assert not isinstance(s_percentages.index, pd.MultiIndex)
            s_percentages *= 100
            trend_type_to_s_percentages[trend_type] = s_percentages
        # Post processing - Add the significant trend into the count of normal trend
        trend_type_to_s_percentages[AbstractTrendTest.POSITIVE_TREND] += trend_type_to_s_percentages[
            AbstractTrendTest.SIGNIFICATIVE_POSITIVE_TREND]
        trend_type_to_s_percentages[AbstractTrendTest.NEGATIVE_TREND] += trend_type_to_s_percentages[
            AbstractTrendTest.SIGNIFICATIVE_NEGATIVE_TREND]
        return trend_type_to_s_percentages

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False):
        def reduction_function_with_level(df_bool):
            if level is None:
                return reduction_function(df_bool)
            else:
                return reduction_function(df_bool, level)
        return {'global': reduction_function_with_level}

    def visualize_trend_test_evolution(self, reduction_function, xlabel, xlabel_values, ax=None, marker='o',
                                       subtitle=''):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.study_visualizer.figsize)

        trend_type_to_percentages_values = {k: s.values for k, s in
                                            self.trend_type_to_s_percentages(reduction_function).items()}
        for trend_type in self.trend_types:
            style = self.trend_type_to_style[trend_type]
            percentages_values = trend_type_to_percentages_values[trend_type]
            ax.plot(xlabel_values, percentages_values, style + marker, label=trend_type)

        # Plot the total value of significative values
        significative_values = trend_type_to_percentages_values[AbstractTrendTest.SIGNIFICATIVE_NEGATIVE_TREND] \
                               + trend_type_to_percentages_values[AbstractTrendTest.SIGNIFICATIVE_POSITIVE_TREND]
        ax.plot(xlabel_values, significative_values, 'y-' + marker, label=AbstractTrendTest.SIGNIFICATIVE + ' trends')

        # Global information
        added_str = 'weighted '
        ylabel = '% averaged on massifs & {}averaged on {}'.format(added_str, xlabel)
        ylabel += ' (with uniform weights)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.set_xticks(xlabel_values)
        ax.set_yticks(list(range(0, 101, 10)))
        ax.grid()
        ax.legend()

        variable_name = self.study.variable_class.NAME
        name = get_display_name_from_object_type(self.trend_test_class)
        title = 'Evolution of {} trends (significative or not) wrt to the {} with {}'.format(variable_name, xlabel,
                                                                                             name)
        title += 'with {} data'.format(subtitle)
        ax.set_title(title)
        self.show_or_save_to_file(specific_title=title)

    def visualize_trend_test_repartition(self, reduction_function, axes=None, subtitle=''):
        if axes is None:
            nb_trend_type = len(self.trend_test_class.trend_type_to_style())
            fig, axes = plt.subplots(1, nb_trend_type, figsize=self.study_visualizer.figsize)

        # Plot weighted percentages over the years
        trend_type_to_s_percentages = self.trend_type_to_s_percentages(reduction_function)
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

    def visualize_year_trend_test(self, ax=None, marker='o', add_detailed_plots=False):
        def year_reduction(df_bool):
            # Take the mean with respect to all the first axis indices
            return df_bool.mean(axis=0)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(year_reduction,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='starting years',
                                                xlabel_values=self.starting_years, ax=ax, marker=marker,
                                                subtitle=subtitle)

    def visualize_altitude_trend_test(self, ax=None, marker='o', add_detailed_plots=False):
        def altitude_reduction(df_bool, level):
            # Take the mean with respect to the years
            df_bool = df_bool.mean(axis=1)
            # Take the mean with respect the massifs
            print(df_bool.head())
            return df_bool.mean(level=level)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(altitude_reduction, level=self.altitude_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_evolution(reduction_function=reduction_function, xlabel='altitude',
                                                xlabel_values=self.altitudes, ax=ax, marker=marker,
                                                subtitle=subtitle)

    def visualize_massif_trend_test(self, axes=None, add_detailed_plots=False):
        def massif_reduction(df_bool, level):
            # Take the mean with respect to the years
            df_bool = df_bool.mean(axis=1)
            # Take the mean with respect the altitude
            return df_bool.mean(level=level)

        for subtitle, reduction_function in self.subtitle_to_reduction_function(massif_reduction,level=self.massif_index_level,
                                                                                add_detailed_plot=add_detailed_plots).items():
            self.visualize_trend_test_repartition(reduction_function, axes, subtitle=subtitle)


class QuantityAltitudeHypercubeVisualizer(AltitudeHypercubeVisualizer):

    @property
    def study_title(self):
        return 'Quantity Altitude Study'

    def subtitle_to_reduction_function(self, reduction_function, level=None, add_detailed_plot=False):
        subtitle_to_reduction_function = super().subtitle_to_reduction_function(reduction_function, level, add_detailed_plot)

        def get_function_from_tuple(tuple_for_axis_0):
            def f(df_bool: pd.DataFrame):
                # Loc with a tuple with respect the axis 0
                df_bool = df_bool.loc[tuple_for_axis_0, :].copy()
                # Apply the reduction function
                if level is None:
                    return reduction_function(df_bool)
                else:
                    return reduction_function(df_bool, level-1)

            return f

        # Add the detailed plot, taken by loc with respect to the first index
        if add_detailed_plot:
            tuples_axis_0 = self.tuple_values(idx=0)
            for tuple_axis_0 in tuples_axis_0:
                subtitle_to_reduction_function[tuple_axis_0] = get_function_from_tuple(tuple_axis_0)
        return subtitle_to_reduction_function

    @property
    def quantities(self):
        return self.tuple_values(idx=0)

    @property
    def altitudes(self):
        return self.tuple_values(idx=1)

    @property
    def altitude_index_level(self):
        return 1

    @property
    def massif_index_level(self):
        return 2
