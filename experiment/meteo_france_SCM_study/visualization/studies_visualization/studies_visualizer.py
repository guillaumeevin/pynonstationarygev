from collections import OrderedDict, Counter
import os
import os.path as op
from multiprocessing.dummy import Pool
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.visualization.studies_visualization.studies import \
    Studies
from experiment.meteo_france_SCM_study.visualization.study_visualization.study_visualizer import StudyVisualizer
from experiment.meteo_france_SCM_study.visualization.utils import plot_df
from utils import cached_property, get_display_name_from_object_type, VERSION_TIME


class StudiesVisualizer(object):

    def __init__(self, studies: Studies) -> None:
        self.studies = studies

    @property
    def first_study(self):
        return self.studies.first_study

    def mean_as_a_function_of_altitude(self, region_only=False):
        # Load the massif names to display
        if region_only:
            assert isinstance(self.first_study, AbstractExtendedStudy)
            massif_names = self.first_study.region_names
        else:
            massif_names = self.first_study.study_massif_names
        # Load the dictionary that maps each massif_name to its corresponding time series
        mean_series = []
        for study in self.studies.altitude_to_study.values():
            mean_serie = study.df_annual_total.loc[:, massif_names].mean(axis=0)
            mean_series.append(mean_serie)
        df_mean = pd.concat(mean_series, axis=1)  # type: pd.DataFrame
        df_mean.columns = self.studies.altitude_list
        plot_df(df_mean)


def get_percentages(v):
    return v.percentages_of_negative_trends()[0]


class AltitudeVisualizer(object):

    def __init__(self, altitude_to_study_visualizer: Dict[int, StudyVisualizer], multiprocessing=False,
                 save_to_file=False):
        self.save_to_file = save_to_file
        self.multiprocessing = multiprocessing
        assert isinstance(altitude_to_study_visualizer, OrderedDict)
        self.altitude_to_study_visualizer = altitude_to_study_visualizer

    @property
    def altitudes(self):
        return list(self.altitude_to_study_visualizer.keys())

    @cached_property
    def all_percentages(self):
        if self.multiprocessing:
            with Pool(4) as p:
                l = p.map(get_percentages, list(self.altitude_to_study_visualizer.values()))
        else:
            l = [get_percentages(v) for v in self.altitude_to_study_visualizer.values()]
        return l

    @property
    def any_study_visualizer(self) -> StudyVisualizer:
        return list(self.altitude_to_study_visualizer.values())[0]

    @property
    def study(self):
        return self.any_study_visualizer.study

    def get_item_fct(self, year):
        idx = self.any_study_visualizer.starting_years.index(year)
        f = lambda s: s[idx]
        return f

    @cached_property
    def starting_year(self):
        return self.any_study_visualizer.starting_years[0]

    def get_top_potential_years(self, reverse=False):
        top_n = 5
        top_top = 3
        # keep the top_n for each altitude
        all_years = [[year for year, _ in sorted(enumerate(p), key=lambda s:s[1], reverse=reverse)[-top_n:]] for p in self.all_percentages]
        from itertools import chain
        all_years = list(chain(*all_years))
        years = [y for y, _ in sorted(Counter(all_years).items(), key=lambda s:s[1])[-top_top:]]
        years = [y + self.starting_year for y in years]
        return years

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

    def negative_trend_percentages_evolution(self, reverse=True):
        curve_name__metric_and_color = [
            ('max', np.max, 'g'),
            ('mean', np.mean, 'b'),
            ('median', np.median, 'c'),
            ('min', np.min, 'r'),
        ]
        # Add some years
        # spotted_years = [1963, 1976]
        # years_to_display = spotted_years
        str_markers = ['o'] + [m for m in Line2D.markers if isinstance(m, str)][3:]
        # for year, marker in zip(years_to_display, str_markers):
        #     new = (str(year), self.get_item_fct(year), 'y', marker + ':')
        #     curve_name__metric_and_color.append(new)

        for year, marker in zip(self.get_top_potential_years(), str_markers):
            new = (str(year), self.get_item_fct(year), 'y', marker + ':')
            curve_name__metric_and_color.append(new)
        for year, marker in zip(self.get_top_potential_years(reverse=True), str_markers):
            new = (str(year), self.get_item_fct(year), 'm', marker + ':')
            curve_name__metric_and_color.append(new)

        fig, ax = plt.subplots(1, 1, figsize=self.any_study_visualizer.figsize)
        for curve_name, metric, color, *marker in curve_name__metric_and_color[:]:
            marker, curve_name = (marker[0], curve_name + ' starting year') if marker \
                else ('-', curve_name + ' over the starting years')
            values = [metric(p) for p in self.all_percentages]
            if reverse:
                values = [100 - v for v in values]
                k = ['max', 'min']
                for before, new in zip(k, k[::-1]):
                    if before in curve_name:
                        curve_name = curve_name.replace(before, new)
                        break
            ax.plot(self.altitudes, values, color + marker, label=curve_name)
        ax.legend()
        ax.set_xticks(self.altitudes)
        ax.set_yticks(list(range(0,101, 10)))
        ax.grid()

        ax.axhline(y=50, color='k')
        word = 'positive' if reverse else 'negative'
        ax.set_ylabel('% of massifs with {} trends'.format(word))
        ax.set_xlabel('altitude')
        variable_name = self.study.variable_class.NAME
        score_name = get_display_name_from_object_type(self.any_study_visualizer.score_class)
        title = 'Evolution of {} trends wrt to the altitude with {}'.format(variable_name, score_name)
        ax.set_title(title)
        self.show_or_save_to_file(specific_title=title)