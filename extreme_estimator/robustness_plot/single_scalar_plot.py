from extreme_estimator.robustness_plot.abstract_robustness_plot import AbstractPlot
import matplotlib.pyplot as plt
from itertools import product


class SingleScalarPlot(AbstractPlot):
    """
    For a single scalar plot, for the combination of all the parameters of interest,
    then the function
    """

    def single_scalar_from_all_params(self, **kwargs_single_point) -> float:
        pass

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
            # todo: do some parallzlization here (do the parallelization in the Asbtract class if possible)
            for plot_row_value in plot_row_values:
                # Adapt the kwargs for the single value
                kwargs_single_point = kwargs_single_plot.copy()
                kwargs_single_point.update({self.plot_row_item.argument_name: plot_row_value,
                                            self.plot_label_item.argument_name: plot_label_value})
                plot_row_value_to_error[plot_row_value] = self.single_scalar_from_all_params(**kwargs_single_point)
            plot_column_values = [plot_row_value_to_error[plot_row_value] for plot_row_value in plot_row_values]
            ax.plot(plot_row_values, plot_column_values, color=self.COLORS[j % len(self.COLORS)], label=str(j))
        ax.legend()
        ax.set_xlabel(self.plot_row_item.dislay_name)
        ax.set_ylabel('Absolute error')
        ax.set_title('Title (display all the other parameters)')
