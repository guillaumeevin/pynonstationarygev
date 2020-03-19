from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.visualization.utils import create_adjusted_axes
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_data.meteo_france_data.scm_models_data.visualization.create_shifted_cmap import imshow_shifted
from extreme_fit.function.abstract_function import AbstractFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.split import Split
from root_utils import cached_property


class AbstractMarginFunction(AbstractFunction):
    """
    AbstractMarginFunction maps points from a space S (could be 1D, 2D,...) to R^3 (the 3 parameters of the GEV)
    """
    VISUALIZATION_RESOLUTION = 100
    VISUALIZATION_TEMPORAL_STEPS = 2

    def __init__(self, coordinates: AbstractCoordinates):
        super().__init__(coordinates)
        self.mask_2D = None

        # Visualization parameters
        self.visualization_axes = None
        self.datapoint_display = False
        self.spatio_temporal_split = Split.all
        self.datapoint_marker = 'o'
        self.color = 'skyblue'
        self.filter = None
        self.linewidth = 1
        self.subplot_space = 1.0

        self.temporal_step_to_grid_2D = {}
        self._grid_1D = None
        self.title = None
        self.add_future_temporal_steps = False

        # Visualization limits
        self._visualization_x_limits = None
        self._visualization_y_limits = None

    @property
    def x(self):
        return self.coordinates.x_coordinates

    @property
    def y(self):
        return self.coordinates.y_coordinates

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Main method that maps each coordinate to its GEV parameters"""
        raise NotImplementedError

    @property
    def gev_value_name_to_serie(self) -> Dict[str, pd.Series]:
        # Load the gev_params
        gev_params = [self.get_gev_params(coordinate) for coordinate in self.coordinates.coordinates_values()]
        # Load the dictionary of values (distribution parameters + the quantiles)
        value_dicts = [gev_param.summary_dict for gev_param in gev_params]
        gev_value_name_to_serie = {}
        for value_name in GevParams.SUMMARY_NAMES:
            s = pd.Series(data=[d[value_name] for d in value_dicts], index=self.coordinates.index)
            gev_value_name_to_serie[value_name] = s
        return gev_value_name_to_serie

    # Visualization function

    def set_datapoint_display_parameters(self, spatio_temporal_split=Split.all, datapoint_marker=None, filter=None,
                                         color=None,
                                         linewidth=1, datapoint_display=False):
        self.datapoint_display = datapoint_display
        self.spatio_temporal_split = spatio_temporal_split
        self.datapoint_marker = datapoint_marker
        self.linewidth = linewidth
        self.filter = filter
        self.color = color

    def visualize_function(self, axes=None, show=True, dot_display=False, title=None):
        self.title = title
        self.datapoint_display = dot_display
        if axes is None:
            if self.coordinates.has_temporal_coordinates:
                axes = create_adjusted_axes(GevParams.NB_SUMMARY_NAMES, self.VISUALIZATION_TEMPORAL_STEPS)
            else:
                axes = create_adjusted_axes(1, GevParams.NB_SUMMARY_NAMES, subplot_space=self.subplot_space)
        self.visualization_axes = axes
        assert len(axes) == GevParams.NB_SUMMARY_NAMES
        for ax, gev_value_name in zip(axes, GevParams.SUMMARY_NAMES):
            self.visualize_single_param(gev_value_name, ax, show=False)
            self.set_title(ax, gev_value_name)
        if show:
            plt.show()
        return axes

    def set_title(self, ax, gev_value_name):
        if hasattr(ax, 'set_title'):
            title_str = gev_value_name if self.title is None else self.title
            ax.set_title(title_str)

    def visualize_single_param(self, gev_value_name=GevParams.LOC, ax=None, show=True):
        assert gev_value_name in GevParams.SUMMARY_NAMES
        nb_coordinates_spatial = self.coordinates.nb_spatial_coordinates
        has_temporal_coordinates = self.coordinates.has_temporal_coordinates
        if nb_coordinates_spatial == 1 and not has_temporal_coordinates:
            self.visualize_1D(gev_value_name, ax, show)
        elif nb_coordinates_spatial == 2 and not has_temporal_coordinates:
            self.visualize_2D(gev_value_name, ax, show)
        elif nb_coordinates_spatial == 2 and has_temporal_coordinates:
            self.visualize_2D_spatial_1D_temporal(gev_value_name, ax, show)
        else:
            raise NotImplementedError('Other visualization not yet implemented')

    # Visualization 1D

    def visualize_1D(self, gev_value_name=GevParams.LOC, ax=None, show=True):
        x = self.coordinates.x_coordinates
        grid, linspace = self.grid_1D(x)
        if ax is None:
            ax = plt.gca()
        if self.datapoint_display:
            ax.plot(linspace, grid[gev_value_name], marker=self.datapoint_marker, color=self.color)
        else:
            ax.plot(linspace, grid[gev_value_name], color=self.color, linewidth=self.linewidth)
        # X axis
        ax.set_xlabel('coordinate X')
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.xaxis.set_tick_params(labelbottom=True)

        if show:
            plt.show()

    def grid_1D(self, x):
        # if self._grid_1D is None:
        #     self._grid_1D = self.get_grid_values_1D(x)
        # return self._grid_1D
        return self.get_grid_values_1D(x, self.spatio_temporal_split)

    def get_grid_values_1D(self, x, spatio_temporal_split):
        # TODO: to avoid getting the value several times, I could cache the results
        if self.datapoint_display:
            # todo: keep only the index of interest here
            linspace = self.coordinates.coordinates_values(spatio_temporal_split)[:, 0]
            if self.filter is not None:
                linspace = linspace[self.filter]
            resolution = len(linspace)
        else:
            resolution = 100
            linspace = np.linspace(x.min(), x.max(), resolution)

        grid = []
        for i, xi in enumerate(linspace):
            gev_param = self.get_gev_params(np.array([xi]))
            assert not gev_param.has_undefined_parameters, 'This case needs to be handled during display,' \
                                                           'gev_parameter for xi={} is undefined'.format(xi)
            grid.append(gev_param.summary_dict)
        grid = {gev_param: [g[gev_param] for g in grid] for gev_param in GevParams.SUMMARY_NAMES}
        return grid, linspace

    # Visualization 2D

    def visualize_2D(self, gev_param_name=GevParams.LOC, ax=None, show=True, temporal_step=None):
        if ax is None:
            ax = plt.gca()

        # Special display
        imshow_shifted(ax, gev_param_name, self.grid_2D(temporal_step)[gev_param_name], self.visualization_extend,
                       self.mask_2D)

        # X axis
        ax.set_xlabel('coordinate X')
        plt.setp(ax.get_xticklabels(), visible=True)
        ax.xaxis.set_tick_params(labelbottom=True)
        # Y axis
        ax.set_ylabel('coordinate Y')
        plt.setp(ax.get_yticklabels(), visible=True)
        ax.yaxis.set_tick_params(labelbottom=True)
        # todo: add dot display in 2D
        if show:
            plt.show()

    @property
    def visualization_x_limits(self):
        if self._visualization_x_limits is None:
            return self.x.min(), self.x.max()
        else:
            return self._visualization_x_limits

    @property
    def visualization_y_limits(self):
        if self._visualization_y_limits is None:
            return self.y.min(), self.y.max()
        else:
            return self._visualization_y_limits

    @property
    def visualization_extend(self):
        return self.visualization_x_limits + self.visualization_y_limits

    def grid_2D(self, temporal_step=None):
        # Cache the results
        if temporal_step not in self.temporal_step_to_grid_2D:
            self.temporal_step_to_grid_2D[temporal_step] = self._grid_2D(temporal_step)
        return self.temporal_step_to_grid_2D[temporal_step]

    def _grid_2D(self, temporal_step=None):
        grid = []
        for xi in np.linspace(*self.visualization_x_limits, self.VISUALIZATION_RESOLUTION):
            for yj in np.linspace(*self.visualization_y_limits, self.VISUALIZATION_RESOLUTION):
                # Build spatio temporal coordinate
                coordinate = [xi, yj]
                if temporal_step is not None:
                    coordinate.append(temporal_step)
                grid.append(self.get_gev_params(np.array(coordinate)).summary_dict)
        grid = {value_name: np.array([g[value_name] for g in grid]).reshape(
            [self.VISUALIZATION_RESOLUTION, self.VISUALIZATION_RESOLUTION])
            for value_name in GevParams.SUMMARY_NAMES}
        return grid

    # Visualization 3D

    def visualize_2D_spatial_1D_temporal(self, gev_param_name=GevParams.LOC, axes=None, show=True):
        if axes is None:
            axes = create_adjusted_axes(self.VISUALIZATION_TEMPORAL_STEPS, 1)
        assert len(axes) == self.VISUALIZATION_TEMPORAL_STEPS

        # Build temporal_steps a list of time steps
        assert len(self.temporal_steps) == self.VISUALIZATION_TEMPORAL_STEPS
        for ax, temporal_step in zip(axes, self.temporal_steps):
            self.visualize_2D(gev_param_name, ax, show=False, temporal_step=temporal_step)
            self.set_title(ax, gev_param_name)

        if show:
            plt.show()

    @cached_property
    def temporal_steps(self) -> List[int]:
        future_temporal_steps = [10, 100] if self.add_future_temporal_steps else []
        nb_past_temporal_step = self.VISUALIZATION_TEMPORAL_STEPS - len(future_temporal_steps)
        start, stop = self.coordinates.df_temporal_range()
        temporal_steps = [int(step) for step in np.linspace(start, stop, num=nb_past_temporal_step)]
        temporal_steps += [stop + step for step in future_temporal_steps]
        return temporal_steps
