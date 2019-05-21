import os
import os.path as op
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
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

    @cached_property
    def df_hypercube(self) -> pd.DataFrame:
        keys = list(self.tuple_to_df_trend_type.keys())
        values = list(self.tuple_to_df_trend_type.values())
        df = pd.concat(values, keys=keys, axis=0)
        return df

    # Some properties

    def show_or_save_to_file(self, specific_title=''):
        if self.save_to_file:
            main_title, _ = '_'.join(self.study.title.split()).split('/')
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
        return list(self.tuple_to_study_visualizer.keys())

    def visualize_altitude_trend_test(self, ax=None, marker='o'):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=self.study_visualizer.figsize)

        # Plot weighted percentages over the years
        for trend_type, style in self.trend_test_class.trend_type_to_style().items():
            altitude_percentages = (self.df_hypercube == trend_type)
            # Take the mean with respect to the years
            altitude_percentages = altitude_percentages.mean(axis=1)
            # Take the mean with respect the massifs
            altitude_percentages = altitude_percentages.mean(axis=0, level=0)
            # Take the numpy array
            altitude_percentages = altitude_percentages.values * 100
            # Plot
            ax.plot(self.altitudes, altitude_percentages, style + marker, label=trend_type)

        # Global information
        added_str = 'weighted '
        ylabel = '% averaged on massifs & {}averaged on starting years'.format(added_str)
        ylabel += ' (with uniform weights)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('altitude')
        ax.set_xticks(self.altitudes)
        ax.set_yticks(list(range(0, 101, 10)))
        ax.grid()
        ax.legend()

        variable_name = self.study.variable_class.NAME
        name = get_display_name_from_object_type(self.trend_test_class)
        title = 'Evolution of {} trends (significative or not) wrt to the altitude with {}'.format(variable_name,name)
        ax.set_title(title)
        self.show_or_save_to_file(specific_title=title)

    def visualize_spatial_trend_test(self, axes=None):
        if axes is None:
            nb_trend_type = len(self.trend_test_class.trend_type_to_style())
            fig, axes = plt.subplots(1, nb_trend_type, figsize=self.study_visualizer.figsize)

        # Plot weighted percentages over the years
        for ax, (trend_type, style) in zip(axes, self.trend_test_class.trend_type_to_style().items()):
            spatial_percentages = (self.df_hypercube == trend_type)
            # Take the mean with respect to the years
            spatial_percentages = spatial_percentages.mean(axis=1)
            # Take the mean with respect the altitude
            spatial_percentages = spatial_percentages.mean(axis=0, level=1) * 100
            # Plot values
            massif_to_value = dict(spatial_percentages)
            cmap = self.trend_test_class.get_cmap_from_trend_type(trend_type)
            self.study.visualize_study(ax, massif_to_value, show=False, cmap=cmap, label=None)
            ax.set_title(trend_type)

        # Global information
        name = get_display_name_from_object_type(self.trend_test_class)
        title = 'Repartition of trends (significative or not) with {}'.format(name)
        title +=  '\n(in % averaged on altitudes & averaged on starting years)'
        StudyVisualizer.clean_axes_write_title_on_the_left(axes, title, left_border=None)
        plt.suptitle(title)
        self.show_or_save_to_file(specific_title=title)


class QuantitityAltitudeHypercubeVisualizer(HypercubeVisualizer):
    pass
