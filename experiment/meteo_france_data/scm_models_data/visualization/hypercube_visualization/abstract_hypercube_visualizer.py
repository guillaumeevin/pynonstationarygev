import os
import os.path as op
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd

from experiment.meteo_france_data.scm_models_data.visualization.study_visualization.study_visualizer import \
    StudyVisualizer
from root_utils import cached_property, VERSION_TIME, get_display_name_from_object_type


class AbstractHypercubeVisualizer(object):
    """
    A study visualizer contain some massifs and years. This forms the base DataFrame of the hypercube
    Additional index will come from the tuple.
    Tuple could contain altitudes, type of snow quantity
    """

    def __init__(self, tuple_to_study_visualizer: Dict[Tuple, StudyVisualizer],
                 trend_test_class,
                 nb_data_reduced_for_speed=False,
                 reduce_strength_array=False,
                 save_to_file=False,
                 first_starting_year=None,
                 last_starting_year=None,
                 exact_starting_year=None,
                 verbose=True,
                 sigma_for_best_year=0.0):
        assert sigma_for_best_year >= 0.0
        self.sigma_for_best_year = sigma_for_best_year
        self.reduce_strength_array = reduce_strength_array
        self.verbose = verbose
        self.save_to_file = save_to_file
        self.trend_test_class = trend_test_class
        self.tuple_to_study_visualizer = tuple_to_study_visualizer  # type: Dict[Tuple, StudyVisualizer]

        if isinstance(nb_data_reduced_for_speed, bool):
            self.nb_data_for_fast_mode = 7 if nb_data_reduced_for_speed else None
        else:
            assert isinstance(nb_data_reduced_for_speed, int)
            self.nb_data_for_fast_mode = nb_data_reduced_for_speed

        if exact_starting_year is not None:
            assert first_starting_year is None and last_starting_year is None
            self.first_starting_year, self.last_starting_year = exact_starting_year, exact_starting_year
        else:
            default_first_starting_year, *_, default_last_starting_year = self.all_potential_starting_years
            self.first_starting_year = first_starting_year if first_starting_year is not None else default_first_starting_year
            self.last_starting_year = last_starting_year if last_starting_year is not None else default_last_starting_year
        # Load starting year
        self.starting_years = [year for year in self.all_potential_starting_years
                               if self.first_starting_year <= year <= self.last_starting_year]
        if self.nb_data_for_fast_mode is not None:
            self.starting_years = self.starting_years[:self.nb_data_for_fast_mode]
            self.last_starting_year = self.starting_years[-1]

        if self.verbose:
            print('Hypercube with parameters:')
            print('First starting year: {}, Last starting year: {}'.format(self.first_starting_year,
                                                                           self.last_starting_year))
            print('Starting years:', self.starting_years)
            print('Trend test class:', get_display_name_from_object_type(self.trend_test_class))

    # Main attributes defining the hypercube

    @property
    def trend_test_name(self):
        return get_display_name_from_object_type(self.trend_test_class)

    @property
    def all_potential_starting_years(self):
        return self.study_visualizer.starting_years

    def tuple_values(self, idx):
        return sorted(set([t[idx] if isinstance(t, tuple) else t for t in self.tuple_to_study_visualizer.keys()]))

    @cached_property
    def df_trends_spatio_temporal(self):
        return [study_visualizer.df_trend_spatio_temporal(self.trend_test_class, self.starting_years,
                                                          self.nb_data_for_fast_mode)
                for study_visualizer in self.tuple_to_study_visualizer.values()]

    def _df_hypercube_trend_meta(self, idx) -> pd.DataFrame:
        df_spatio_temporal_trend_strength = [e[idx] for e in self.df_trends_spatio_temporal]
        return pd.concat(df_spatio_temporal_trend_strength, keys=list(self.tuple_to_study_visualizer.keys()), axis=0)

    @cached_property
    def df_hypercube_trend_type(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=0
                                             )

    @cached_property
    def df_hypercube_trend_slope_relative_strength(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=1)

    @cached_property
    def df_hypercube_trend_nllh(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=2)

    @cached_property
    def df_hypercube_trend_constant_quantile(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=3)

    @cached_property
    def df_hypercube_trend_mean_same_sign(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=4)

    @cached_property
    def df_hypercube_trend_variance_same_sign(self) -> pd.DataFrame:
        return self._df_hypercube_trend_meta(idx=5)

    # Some properties

    @property
    def study_title(self):
        return self.study.title

    def show_or_save_to_file(self, specific_title='', tight=False, dpi=None):
        if self.save_to_file:
            main_title, *_ = '_'.join(self.study_title.split()).split('/')
            filename = "{}/{}/".format(VERSION_TIME, main_title)
            filename += specific_title
            filepath = op.join(self.study.result_full_path, filename + '.png')
            dirname = op.dirname(filepath)
            if not op.exists(dirname):
                os.makedirs(dirname, exist_ok=True)
            if tight:
                plt.savefig(filepath, bbox_inches='tight', pad_inches=+0.03, dpi=1000)
            elif dpi is not None:
                plt.savefig(filepath, dpi=dpi)
            else:
                plt.savefig(filepath)
        else:
            plt.show()
        plt.close()

    @property
    def study_visualizer(self) -> StudyVisualizer:
        return list(self.tuple_to_study_visualizer.values())[0]

    @property
    def study(self):
        return self.study_visualizer.study
