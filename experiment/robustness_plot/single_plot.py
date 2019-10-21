import os
import os.path as op

import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from experiment.robustness_plot.display_item import DisplayItem
from root_utils import get_full_path

plt.style.use('seaborn-white')


class SinglePlot(object):
    COLORS = ['blue', 'red', 'green', 'black', 'magenta', 'cyan']
    OrdinateItem = DisplayItem('ordinate', AbstractEstimator.MAE_ERROR)

    def __init__(self, grid_row_item, grid_column_item, plot_row_item, plot_label_item, nb_samples=1, main_title='',
                 plot_png_filename=None):
        self.grid_row_item = grid_row_item  # type: DisplayItem
        self.grid_column_item = grid_column_item  # type: DisplayItem
        self.plot_row_item = plot_row_item  # type: DisplayItem
        self.plot_label_item = plot_label_item  # type: DisplayItem
        self.nb_samples = nb_samples
        self.main_title = main_title
        self.plot_png_filename = plot_png_filename

    def robustness_grid_plot(self, **kwargs):
        # Extract Grid row and columns values
        grid_row_values = self.grid_row_item.values_from_kwargs(**kwargs)
        grid_column_values = self.grid_column_item.values_from_kwargs(**kwargs)
        nb_grid_rows, nb_grid_columns = len(grid_row_values), len(grid_column_values)
        # Start the overall plot
        # fig = plt.figure()
        fig, axes = plt.subplots(nb_grid_rows, nb_grid_columns, sharex='col', sharey='row')
        fig.subplots_adjust(hspace=0.4, wspace=0.4, )
        for (i, grid_row_value), (j, grid_column_value) in product(enumerate(grid_row_values),
                                                                   enumerate(grid_column_values)):
            print('Grid plot: {}={} {}={}'.format(self.grid_row_item.name, grid_row_value,
                                                  self.grid_column_item.name, grid_column_value))
            ax = axes[i, j]
            # ax = fig.add_subplot(nb_grid_rows, nb_grid_columns, i)
            # Adapt the kwargs for the single plot
            kwargs_single_plot = kwargs.copy()
            kwargs_single_plot[self.grid_row_item.name] = grid_row_value
            kwargs_single_plot[self.grid_column_item.name] = grid_column_value
            self.robustness_single_plot(ax, **kwargs_single_plot)
            self.add_sub_title(ax, grid_column_value, grid_row_value)
        fig.suptitle(self.main_title)
        self.save_plot()
        plt.show()

    def save_plot(self):
        if self.plot_png_filename is None:
            return
        assert isinstance(self.plot_png_filename, str)
        relative_path = op.join('local', 'plot')
        plot_pn_dirpath = get_full_path(relative_path=relative_path)
        if not op.exists(plot_pn_dirpath):
            os.makedirs(plot_pn_dirpath)
        plot_pn_filepath = op.join(plot_pn_dirpath, self.plot_png_filename + '.png')
        i = 2
        while op.exists(plot_pn_filepath):
            plot_pn_filepath = op.join(plot_pn_dirpath, self.plot_png_filename + str(i) + '.png')
            i += 1
        # plt.savefig(plot_pn_filepath, bbox_inches='tight')
        plt.savefig(plot_pn_filepath)

    def robustness_single_plot(self, ax, **kwargs_single_plot):
        plot_row_values = self.plot_row_item.values_from_kwargs(**kwargs_single_plot)
        plot_label_values = self.plot_label_item.values_from_kwargs(**kwargs_single_plot)
        ordinate_name = self.OrdinateItem.value_from_kwargs(**kwargs_single_plot)
        for j, plot_label_value in enumerate(plot_label_values):
            mean_values, std_values = self.compute_mean_and_std_ordinate_values(kwargs_single_plot, ordinate_name,
                                                                                plot_label_value, plot_row_values)
            ax.errorbar(plot_row_values, mean_values, std_values,
                        # linestyle='None', marker='^',
                        linewidth=0.5,
                        color=self.COLORS[j % len(self.COLORS)],
                        label=self.plot_label_item.display_name_from_value(plot_label_value))
        ax.legend()
        # X axis
        ax.set_xlabel(self.plot_row_item.name)
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        # Y axis
        ax.set_ylabel(ordinate_name + ' ({} samples)'.format(self.nb_samples))
        plt.setp(ax.get_yticklabels(), visible=True)
        ax.yaxis.set_tick_params(labelbottom=True)

    def compute_mean_and_std_ordinate_values(self, kwargs_single_plot, ordinate_name, plot_label_value,
                                             plot_row_values):
        all_ordinate_values = []
        for nb_sample in range(self.nb_samples):
            # Important to add the nb_sample argument, to differentiate the different experiments
            kwargs_single_plot['nb_sample'] = nb_sample
            ordinate_values = self.compute_ordinate_values(kwargs_single_plot, ordinate_name, plot_label_value,
                                                           plot_row_values)
            all_ordinate_values.append(ordinate_values)
        all_ordinate_values = np.array(all_ordinate_values)
        return np.mean(all_ordinate_values, axis=0), np.std(all_ordinate_values, axis=0)

    def compute_ordinate_values(self, kwargs_single_plot, ordinate_name, plot_label_value, plot_row_values):
        # Compute
        plot_row_value_to_ordinate_value = {}
        # todo: do some parallzlization here
        for plot_row_value in plot_row_values:
            # Adapt the kwargs for the single value
            kwargs_single_point = kwargs_single_plot.copy()
            kwargs_single_point.update({self.plot_row_item.name: plot_row_value,
                                        self.plot_label_item.name: plot_label_value})
            # The kwargs should not contain list of values
            for k, v in kwargs_single_point.items():
                assert not isinstance(v, list), '"{}" argument is a list'.format(k)
            # Compute ordinate values
            ordinate_name_to_ordinate_value = self.cached_compute_value_from_kwargs_single_point(**kwargs_single_point)
            plot_row_value_to_ordinate_value[plot_row_value] = ordinate_name_to_ordinate_value[ordinate_name]
        # Plot the figure
        plot_ordinate_values = [plot_row_value_to_ordinate_value[plot_row_value] for plot_row_value in
                                plot_row_values]
        return plot_ordinate_values

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point) -> dict:
        pass

    def cached_compute_value_from_kwargs_single_point(self, **kwargs_single_point) -> dict:
        return self.compute_value_from_kwargs_single_point(**kwargs_single_point)

    def add_sub_title(self, ax, grid_column_value, grid_row_value):
        title_str = self.grid_row_item.display_name_from_value(grid_row_value)
        title_str += '      ' if len(title_str) > 0 else ''
        title_str += self.grid_column_item.display_name_from_value(grid_column_value)
        ax.set_title(title_str)
