import os
import os
import os.path as op
from multiprocessing.dummy import Pool
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from utils import cached_property, VERSION_TIME


def get_df_trend_spatio_temporal(study_visualizer, trend_class, starting_years):
    return study_visualizer.df_trend_spatio_temporal(trend_class, starting_years)


class HypercubeVisualizer(object):
    """
    A study visualizer contain some massifs and years. This forms the base DataFrame of the hypercube
    Additional index will come from the tuple.
    Tuple could contain altitudes, type of snow quantity
    """

    def __init__(self, tuple_to_study_visualizer: Dict[Tuple, StudyVisualizer],
                 trend_class,
                 save_to_file=False):
        self.save_to_file = save_to_file
        self.trend_class = trend_class
        self.tuple_to_study_visualizer = tuple_to_study_visualizer  # type: Dict[Tuple, StudyVisualizer]

    # Main attributes defining the hypercube

    def tuple_to_massif_names(self, tuple):
        return self.tuple_to_study_visualizer[tuple].study.study_massif_names

    @cached_property
    def starting_years(self):
        return self.study_visualizer.starting_years[:7]

    @property
    def starting_year_to_weights(self):
        # Load uniform weights by default
        uniform_weight = 1 / len(self.starting_years)
        return {year: uniform_weight for year in self.starting_years}

    @cached_property
    def tuple_to_df_trend_type(self):
        df_spatio_temporal_trend_types = [get_df_trend_spatio_temporal(study_visualizer, self.trend_class, self.starting_years)
                                          for study_visualizer in self.tuple_to_study_visualizer.values()]
        return dict(zip(self.tuple_to_study_visualizer.keys(), df_spatio_temporal_trend_types))

    @cached_property
    def hypercube(self):
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


class AltitudeHypercubeVisualizer(HypercubeVisualizer):
    pass


class QuantitityAltitudeHypercubeVisualizer(HypercubeVisualizer):
    pass
