import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.robustness_plot.display_item import DisplayItem

plt.style.use('seaborn-white')


class SinglePlot(object):
    COLORS = ['blue', 'red', 'green', 'black', 'magenta', 'cyan']
    OrdinateItem = DisplayItem('ordinate', AbstractEstimator.MAE_ERROR)

    def __init__(self, grid_row_item, grid_column_item, plot_row_item, plot_label_item, nb_samples=1):
        self.grid_row_item = grid_row_item  # type: DisplayItem
        self.grid_column_item = grid_column_item  # type: DisplayItem
        self.plot_row_item = plot_row_item  # type: DisplayItem
        self.plot_label_item = plot_label_item  # type: DisplayItem
        self.nb_samples = nb_samples

    def robustness_grid_plot(self, **kwargs):
        # Extract Grid row and columns values
        grid_row_values = self.grid_row_item.values_from_kwargs(**kwargs)
        grid_column_values = self.grid_column_item.values_from_kwargs(**kwargs)
        nb_grid_rows, nb_grid_columns = len(grid_row_values), len(grid_column_values)
        # Start the overall plot
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, (grid_row_value, grid_column_value) in enumerate(product(grid_row_values, grid_column_values), 1):
            print('Grid plot: {}={} {}={}'.format(self.grid_row_item.name, grid_row_value,
                                                  self.grid_column_item.name, grid_column_value))
            ax = fig.add_subplot(nb_grid_rows, nb_grid_columns, i)
            # Adapt the kwargs for the single plot
            kwargs_single_plot = kwargs.copy()
            kwargs_single_plot[self.grid_row_item.name] = grid_row_value
            kwargs_single_plot[self.grid_column_item.name] = grid_column_value
            self.robustness_single_plot(ax, **kwargs_single_plot)
            self.add_title(ax, grid_column_value, grid_row_value)
        plt.show()

    def robustness_single_plot(self, ax, **kwargs_single_plot):
        plot_row_values = self.plot_row_item.values_from_kwargs(**kwargs_single_plot)
        plot_label_values = self.plot_label_item.values_from_kwargs(**kwargs_single_plot)
        ordinate_name = self.OrdinateItem.value_from_kwargs(**kwargs_single_plot)
        for j, plot_label_value in enumerate(plot_label_values):
            mean_values, std_values = self.compute_mean_and_std_ordinate_values(kwargs_single_plot, ordinate_name,
                                                                                plot_label_value, plot_row_values)
            ax.errorbar(plot_row_values, mean_values, std_values,
                    # linestyle='None', marker='^',
                    linewidth = 0.5,
                    color=self.COLORS[j % len(self.COLORS)],
                    label=self.plot_label_item.display_name_from_value(plot_label_value))
        ax.legend()
        ax.set_xlabel(self.plot_row_item.name)
        ax.set_ylabel(ordinate_name)

    def compute_mean_and_std_ordinate_values(self, kwargs_single_plot, ordinate_name, plot_label_value, plot_row_values):
        all_ordinate_values = []
        for nb_sample in range(self.nb_samples):
            # Important to add the nb_sample argument, to differentiate the different experiments
            kwargs_single_plot['nb_sample'] = nb_sample
            ordinate_values = self.compute_ordinate_values(kwargs_single_plot, ordinate_name, plot_label_value, plot_row_values)
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
            print(ordinate_name, plot_row_value)
            plot_row_value_to_ordinate_value[plot_row_value] = ordinate_name_to_ordinate_value[ordinate_name]
        # Plot the figure
        plot_ordinate_values = [plot_row_value_to_ordinate_value[plot_row_value] for plot_row_value in
                                plot_row_values]
        return plot_ordinate_values

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point) -> dict:
        pass

    def cached_compute_value_from_kwargs_single_point(self, **kwargs_single_point) -> dict:
        return self.compute_value_from_kwargs_single_point(**kwargs_single_point)

    def add_title(self, ax, grid_column_value, grid_row_value):
        title_str = self.grid_row_item.display_name_from_value(grid_row_value)
        title_str += '      ' if len(title_str) > 0 else ''
        title_str += self.grid_column_item.display_name_from_value(grid_column_value)
        ax.set_title(title_str)
