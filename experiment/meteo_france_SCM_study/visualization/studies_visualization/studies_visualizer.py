from collections import OrderedDict, Counter
import os
import os.path as op
from multiprocessing.dummy import Pool
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.abstract_trend_test import AbstractTrendTest
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
        self.altitude_to_study_visualizer = altitude_to_study_visualizer  # type: Dict[int, StudyVisualizer]

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
        all_years = [[year for year, _ in sorted(enumerate(p), key=lambda s: s[1], reverse=reverse)[-top_n:]] for p in
                     self.all_percentages]
        from itertools import chain
        all_years = list(chain(*all_years))
        years = [y for y, _ in sorted(Counter(all_years).items(), key=lambda s: s[1])[-top_top:]]
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
        ax.set_yticks(list(range(0, 101, 10)))
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

    # Trend tests evolution

    def trend_tests_percentage_evolution(self, trend_test_classes, starting_year_to_weights: None):
        # Load uniform weights by default
        if starting_year_to_weights is None:
            startings_years = self.any_study_visualizer.starting_years
            uniform_weight = 1 / len(startings_years)
            starting_year_to_weights = {year: uniform_weight for year in startings_years}
        else:
            uniform_weight = 0.0

        fig, ax = plt.subplots(1, 1, figsize=self.any_study_visualizer.figsize)

        # Create one display for each trend test class
        markers = ['o', '+']
        assert len(markers) >= len(trend_test_classes)
        # Add a second legend for the color and to explain the line

        for marker, trend_test_class in zip(markers, trend_test_classes):
            self.trend_test_percentages_evolution(ax, marker, trend_test_class, starting_year_to_weights)

        # Add the color legend
        handles, labels = ax.get_legend_handles_labels()
        handles_ax, labels_ax = handles[:5], labels[:5]
        ax.legend(handles_ax, labels_ax, markerscale=0.0, loc=1)
        ax.set_xticks(self.altitudes)
        ax.set_yticks(list(range(0, 101, 10)))
        ax.grid()

        # Add the marker legend
        names = [get_display_name_from_object_type(c) for c in trend_test_classes]
        handles_ax2, labels_ax2 = handles[::5], names
        ax2 = ax.twinx()
        ax2.legend(handles_ax2, labels_ax2, loc=2)
        ax2.set_yticks([])

        # Global information
        added_str = ''if uniform_weight > 0.0 else 'weighted '
        ylabel = '% averaged on massifs & {}averaged on starting years'.format(added_str)
        ylabel += ' (with uniform weights)'
        ax.set_ylabel(ylabel)
        ax.set_xlabel('altitude')
        variable_name = self.study.variable_class.NAME
        title = 'Evolution of {} trends (significative or not) wrt to the altitude with {}'.format(variable_name, ', '.join(names))
        ax.set_title(title)
        self.show_or_save_to_file(specific_title=title)

    def trend_test_percentages_evolution(self, ax, marker, trend_test_class, starting_year_to_weights):
        """
        Positive trend in green
        Negative trend in red
        Non significative trend with dotted line
        Significative trend with real line

        :return:
        """
        # Build OrderedDict mapping altitude to a mean serie
        altitude_to_serie_with_mean_percentages = OrderedDict()
        for altitude, study_visualizer in self.altitude_to_study_visualizer.items():
            s = study_visualizer.serie_mean_trend_test_count(trend_test_class, starting_year_to_weights)
            altitude_to_serie_with_mean_percentages[altitude] = s
        # Plot lines
        for trend_type, style in AbstractTrendTest.TREND_TYPE_TO_STYLE.items():
            percentages = [v.loc[trend_type] if trend_type in v.index else 0.0
                           for v in altitude_to_serie_with_mean_percentages.values()]
            if set(percentages) == {0.0}:
                continue
            else:
                ax.plot(self.altitudes, percentages, style + marker, label=trend_type)
