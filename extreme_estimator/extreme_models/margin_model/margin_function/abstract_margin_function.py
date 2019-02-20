from typing import Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.plot.create_shifted_cmap import plot_extreme_param, imshow_shifted
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.split import Split


class AbstractMarginFunction(object):
    """ Class of function mapping points from a space S (could be 1D, 2D,...) to R^3 (the 3 parameters of the GEV)"""

    def __init__(self, coordinates: AbstractCoordinates):
        self.coordinates = coordinates

        # Visualization parameters
        self.visualization_axes = None
        self.datapoint_display = False
        self.spatio_temporal_split = Split.all
        self.datapoint_marker = 'o'
        self.color = 'skyblue'
        self.filter = None
        self.linewidth = 1

        self._grid_2D = None
        self._grid_1D = None

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Main method that maps each coordinate to its GEV parameters"""
        pass

    @property
    def gev_value_name_to_serie(self) -> Dict[str, pd.Series]:
        # Load the gev_params
        gev_params = [self.get_gev_params(coordinate) for coordinate in self.coordinates.coordinates_values()]
        # Load the dictionary of values (margin_fits parameters + the quantiles)
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
        self.datapoint_display = dot_display
        if axes is None:
            fig, axes = plt.subplots(1, len(GevParams.SUMMARY_NAMES))
            fig.subplots_adjust(hspace=1.0, wspace=1.0)
        self.visualization_axes = axes
        for i, gev_value_name in enumerate(GevParams.SUMMARY_NAMES):
            ax = axes[i]
            self.visualize_single_param(gev_value_name, ax, show=False)
            title_str = gev_value_name if title is None else title
            ax.set_title(title_str)
        if show:
            plt.show()
        return axes

    def visualize_single_param(self, gev_value_name=GevParams.LOC, ax=None, show=True):
        assert gev_value_name in GevParams.SUMMARY_NAMES
        if self.coordinates.nb_coordinates_spatial == 1:
            self.visualize_1D(gev_value_name, ax, show)
        elif self.coordinates.nb_coordinates_spatial == 2:
            self.visualize_2D(gev_value_name, ax, show)
        else:
            raise NotImplementedError('3D Margin visualization not yet implemented')

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
            grid.append(gev_param.summary_dict)
        grid = {gev_param: [g[gev_param] for g in grid] for gev_param in GevParams.SUMMARY_NAMES}
        return grid, linspace

    # Visualization 2D

    def visualize_2D(self, gev_param_name=GevParams.LOC, ax=None, show=True):
        x = self.coordinates.x_coordinates
        y = self.coordinates.y_coordinates
        grid = self.grid_2D(x, y)
        if ax is None:
            ax = plt.gca()

        # Special display
        imshow_shifted(ax, gev_param_name, grid[gev_param_name], x, y)

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

    def grid_2D(self, x, y):
        resolution = 100
        grid = []
        for i, xi in enumerate(np.linspace(x.min(), x.max(), resolution)):
            for j, yj in enumerate(np.linspace(y.min(), y.max(), resolution)):
                grid.append(self.get_gev_params(np.array([xi, yj])).summary_dict)
        grid = {value_name: np.array([g[value_name] for g in grid]).reshape([resolution, resolution])
                for value_name in GevParams.SUMMARY_NAMES}
        return grid
