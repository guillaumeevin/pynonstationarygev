from typing import List

from extreme_estimator.estimator.msp_estimator import MaxStableEstimator
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import GaussianMSP, MaxStableModel
from itertools import product

from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinate import CircleCoordinates
import matplotlib.pyplot as plt

plt.style.use('seaborn-white')


class DisplayItem(object):

    def __init__(self, argument_name, default_value, dislay_name=None):
        self.argument_name = argument_name
        self.default_value = default_value
        self.dislay_name = dislay_name if dislay_name is not None else self.argument_name

    def values_from_kwargs(self, **kwargs):
        return kwargs.get(self.argument_name, [self.default_value])

    def value_from_kwargs(self, **kwargs):
        return kwargs.get(self.argument_name, self.default_value)


MaxStableModelItem = DisplayItem('max_stable_model', GaussianMSP)
SpatialCoordinateClassItem = DisplayItem('spatial_coordinate_class', CircleCoordinates)
SpatialParamsItem = DisplayItem('spatial_params', {"r": 1})
NbStationItem = DisplayItem('nb_station', None)
NbObservationItem = DisplayItem('nb_obs', 50)


class AbstractRobustnessPlot(object):

    def __init__(self, grid_row_item, grid_column_item, plot_row_item, plot_label_item):
        self.grid_row_item = grid_row_item  # type: DisplayItem
        self.grid_column_item = grid_column_item  # type: DisplayItem
        self.plot_row_item = plot_row_item  # type: DisplayItem
        self.plot_label_item = plot_label_item  # type: DisplayItem

        self.estimation_error = self.estimation_error_max_stable_unitary_frechet

    def robustness_grid_plot(self, **kwargs):
        grid_row_values = self.grid_row_item.values_from_kwargs(**kwargs)
        grid_column_values = self.grid_column_item.values_from_kwargs(**kwargs)
        nb_grid_rows, nb_grid_columns = len(grid_row_values), len(grid_column_values)
        fig = plt.figure()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i, (grid_row_value, grid_column_value) in enumerate(product(grid_row_values, grid_column_values), 1):
            print('Grid plot: {}={} {}={}'.format(self.grid_row_item.dislay_name, grid_row_value,
                                                  self.grid_column_item.dislay_name, grid_column_value))
            ax = fig.add_subplot(nb_grid_rows, nb_grid_columns, i)
            kwargs_single_plot = kwargs.copy()
            kwargs_single_plot.update({self.grid_row_item.argument_name: grid_row_value,
                                       self.grid_column_item.argument_name: grid_column_value})
            self.robustness_single_plot(ax, **kwargs_single_plot)
        plt.show()

    def robustness_single_plot(self, ax, **kwargs_single_plot):
        plot_row_values = self.plot_row_item.values_from_kwargs(**kwargs_single_plot)
        plot_label_values = self.plot_label_item.values_from_kwargs(**kwargs_single_plot)
        colors = ['blue', 'red', 'green', 'black']
        assert isinstance(plot_label_values, list), plot_label_values
        assert isinstance(plot_row_values, list), plot_row_values
        for j, plot_label_value in enumerate(plot_label_values):
            plot_row_value_to_error = {}
            # todo: do some parallzlization here
            for plot_row_value in plot_row_values:
                kwargs_single_point = kwargs_single_plot.copy()
                kwargs_single_point.update({self.plot_row_item.argument_name: plot_row_value,
                                            self.plot_label_item.argument_name: plot_label_value})
                plot_row_value_to_error[plot_row_value] = self.estimation_error(**kwargs_single_point)
            plot_column_values = [plot_row_value_to_error[plot_row_value] for plot_row_value in plot_row_values]
            ax.plot(plot_row_values, plot_column_values, color=colors[j % len(colors)], label=str(j))
        ax.legend()
        ax.set_xlabel(self.plot_row_item.dislay_name)
        ax.set_ylabel('Absolute error')
        ax.set_title('Title (display all the other parameters)')

    @staticmethod
    def estimation_error_max_stable_unitary_frechet(**kwargs_single_points):
        # Get the argument from kwargs
        print(kwargs_single_points)
        max_stable_model = MaxStableModelItem.value_from_kwargs(**kwargs_single_points)
        spatial_coordinate_class = SpatialCoordinateClassItem.value_from_kwargs(**kwargs_single_points)
        nb_station = NbStationItem.value_from_kwargs(**kwargs_single_points)
        spatial_params = SpatialParamsItem.value_from_kwargs(**kwargs_single_points)
        nb_obs = NbObservationItem.value_from_kwargs(**kwargs_single_points)
        # Run the estimation
        spatial_coordinate = spatial_coordinate_class.from_nb_points(nb_points=nb_station, **spatial_params)
        dataset = SimulatedDataset.from_max_stable_sampling(nb_obs=nb_obs, max_stable_model=max_stable_model,
                                                            spatial_coordinates=spatial_coordinate)
        estimator = MaxStableEstimator(dataset, max_stable_model)
        estimator.fit()
        errors = estimator.error(max_stable_model.params_sample)
        return errors
