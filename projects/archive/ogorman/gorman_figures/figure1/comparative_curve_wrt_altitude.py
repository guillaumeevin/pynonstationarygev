from collections import OrderedDict
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

from cached_property import cached_property

from projects.archive.ogorman.gorman_figures import \
    StudyVisualizerForReturnLevelChange


class ComparativeCurveWrtAltitude(object):

    def __init__(self, altitude_to_study_visualizer):
        assert isinstance(altitude_to_study_visualizer, OrderedDict)
        self.altitude_to_study_visualizer = altitude_to_study_visualizer  # type: OrderedDict[int, StudyVisualizerForReturnLevelChange]

    @cached_property
    def visualizer(self):
        return list(self.altitude_to_study_visualizer.values())[0]

    @property
    def altitudes(self):
        return list(self.altitude_to_study_visualizer.keys())

    def all_plots(self):
        self.shape_shoe_plot()
        self.return_level_shot_plot()

    def shape_shoe_plot(self):
        altitude_to_list_couple = OrderedDict()
        for a, v in self.altitude_to_study_visualizer.items():
            altitude_to_list_couple[a] = v.result_from_double_stationary_fit.shape_list_couple
        label = 'shape parameter'
        self.abstract_shoe_plot(altitude_to_list_couple, plot_name=label, ylabel=label, relative_change=False)

    def return_level_shot_plot(self):
        altitude_to_list_couple = OrderedDict()
        for a, v in self.altitude_to_study_visualizer.items():
            altitude_to_list_couple[a] = v.result_from_double_stationary_fit.return_level_list_couple
        label = '{}-year return level ({})'.format(self.visualizer.return_period, self.visualizer.study.variable_unit)
        self.abstract_shoe_plot(altitude_to_list_couple, plot_name='return level', ylabel=label)

    def abstract_shoe_plot(self, altitude_to_list_couple, plot_name, ylabel, relative_change=True):
        # Prepare axis
        ax = plt.gca()
        ax2 = ax.twinx()

        # Prepare data
        altitude_to_all_relative_differences = OrderedDict()
        altitude_to_all_differences = OrderedDict()
        altitude_to_mean_value_before = OrderedDict()
        altitude_to_mean_value_after = OrderedDict()
        for altitude in self.altitudes:
            list_couple = altitude_to_list_couple[altitude]
            before, after = [np.array(a) for a in zip(*list_couple)]
            altitude_to_mean_value_before[altitude] = before.mean()
            altitude_to_mean_value_after[altitude] = after.mean()
            altitude_to_all_relative_differences[altitude] = 100 * (after - before) / before
            altitude_to_all_differences[altitude] = after - before

        # ax2: Boxplot

        width = 100
        x = altitude_to_all_relative_differences if relative_change else altitude_to_all_differences
        ax2.boxplot(x.values(), positions=self.altitudes, widths=width)
        ax2.set_xlim([min(self.altitudes) - width, max(self.altitudes) + width])
        if relative_change:
            ylabel2 = 'relative change in {} (%)'.format((ylabel.split('(')[0]))
        else:
            ylabel2 = 'change in {}'.format(ylabel)
        ax2.set_ylabel('Distribution of ' + ylabel2)

        # ax: Mean plots on top
        values = [list(v.values()) for v in [altitude_to_mean_value_before, altitude_to_mean_value_after]]
        labels = [
            '{}-{}'.format(self.visualizer.year_min_before, self.visualizer.year_max_before),
            '{}-{}'.format(self.visualizer.year_min_after, self.visualizer.year_max_after),
        ]
        for label, value in zip(labels, values):
            ax.plot(self.altitudes, value, label=label, marker='o')
        ax.set_ylabel('Mean {}'.format(ylabel))
        ax.legend(loc='upper left')  # , prop={'size': size})
        ax.set_xlabel('Altitude (m)')

        # Set same limits to align axis
        a = list(x.values())
        a = list(chain.from_iterable(a))
        all_values = a + values[0] + values[1]
        epsilon = 0.03 * (max(all_values) - min(all_values))
        ylim = [min(all_values) - epsilon, max(all_values) + epsilon]
        ax.set_ylim(ylim)
        ax2.set_ylim(ylim)
        ax.grid()

        self.visualizer.plot_name = plot_name + ' summary'
        self.visualizer.show_or_save_to_file(add_classic_title=False, tight_layout=True, no_title=True,
                                             dpi=500)
        plt.close()
