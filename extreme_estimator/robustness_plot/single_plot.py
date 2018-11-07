import matplotlib.pyplot as plt
from itertools import product

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.robustness_plot.display_item import DisplayItem

plt.style.use('seaborn-white')


class SinglePlot(object):
    COLORS = ['blue', 'red', 'green', 'black', 'magenta', 'cyan']
    OrdinateItem = DisplayItem('ordinate', AbstractEstimator.MAE_ERROR)

    def __init__(self, grid_row_item, grid_column_item, plot_row_item, plot_label_item):
        self.grid_row_item = grid_row_item  # type: DisplayItem
        self.grid_column_item = grid_column_item  # type: DisplayItem
        self.plot_row_item = plot_row_item  # type: DisplayItem
        self.plot_label_item = plot_label_item  # type: DisplayItem

    def robustness_grid_plot(self, **kwargs):
        # Extract Grid row and columns values
        grid_row_values = self.grid_row_item.values_from_kwargs(**kwargs)
        grid_column_values = self.grid_column_item.values_from_kwargs(**kwargs)
        nb_grid_rows, nb_grid_columns = len(grid_row_values), len(grid_column_values)
        # Start the overall plot
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, (grid_row_value, grid_column_value) in enumerate(product(grid_row_values, grid_column_values), 1):
            print('Grid plot: {}={} {}={}'.format(self.grid_row_item.dislay_name, grid_row_value,
                                                  self.grid_column_item.dislay_name, grid_column_value))
            ax = fig.add_subplot(nb_grid_rows, nb_grid_columns, i)
            # Adapt the kwargs for the single plot
            kwargs_single_plot = kwargs.copy()
            kwargs_single_plot.update({self.grid_row_item.argument_name: grid_row_value,
                                       self.grid_column_item.argument_name: grid_column_value})
            self.robustness_single_plot(ax, **kwargs_single_plot)
        plt.show()

    def robustness_single_plot(self, ax, **kwargs_single_plot):
        plot_row_values = self.plot_row_item.values_from_kwargs(**kwargs_single_plot)
        plot_label_values = self.plot_label_item.values_from_kwargs(**kwargs_single_plot)
        for j, plot_label_value in enumerate(plot_label_values):
            # Compute
            plot_row_value_to_error = {}
            # todo: do some parallzlization here
            for plot_row_value in plot_row_values:
                # Adapt the kwargs for the single value
                kwargs_single_point = kwargs_single_plot.copy()
                kwargs_single_point.update({self.plot_row_item.argument_name: plot_row_value,
                                            self.plot_label_item.argument_name: plot_label_value})
                # The kwargs should not contain list of values
                for k, v in kwargs_single_point.items():
                    assert not isinstance(v, list), '"{}" argument is a list'.format(k)
                # Compute ordinate values
                ordinates = self.compute_value_from_kwargs_single_point(**kwargs_single_point)
                # Extract the ordinate value of interest
                ordinate_name = self.OrdinateItem.value_from_kwargs(**kwargs_single_point)
                plot_row_value_to_error[plot_row_value] = ordinates[ordinate_name]
            plot_column_values = [plot_row_value_to_error[plot_row_value] for plot_row_value in plot_row_values]
            ax.plot(plot_row_values, plot_column_values, color=self.COLORS[j % len(self.COLORS)], label=str(j))
        ax.legend()
        ax.set_xlabel(self.plot_row_item.dislay_name)
        ax.set_ylabel('Absolute error')
        ax.set_title('Title (display all the other parameters)')

    def compute_value_from_kwargs_single_point(self, **kwargs_single_point):
        pass
