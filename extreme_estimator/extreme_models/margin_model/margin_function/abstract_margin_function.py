import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class AbstractMarginFunction(object):
    """ Class of function mapping points from a space S (could be 1D, 2D,...) to R^3 (the 3 parameters of the GEV)"""

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams):
        self.spatial_coordinates = spatial_coordinates
        self.default_params = default_params.to_dict()

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Main function that maps each coordinate to its GEV parameters"""
        pass

    # Visualization function

    def visualize_2D(self, gev_param_name=GevParams.GEV_LOC, ax=None, show=False):
        x = self.spatial_coordinates.x_coordinates
        y = self.spatial_coordinates.y_coordinates
        grid = self.get_grid_2D(x, y)
        gev_param_idx = GevParams.GEV_PARAM_NAMES.index(gev_param_name)
        if ax is None:
            ax = plt.gca()
        imshow_method = ax.imshow
        imshow_method(grid[..., gev_param_idx], extent=(x.min(), x.max(), y.min(), y.max()),
                      interpolation='nearest', cmap=cm.gist_rainbow)
        if show:
            plt.show()

    def get_grid_2D(self, x, y):
        resolution = 100
        grid = np.zeros([resolution, resolution, 3])
        for i, xi in enumerate(np.linspace(x.min(), x.max(), resolution)):
            for j, yj in enumerate(np.linspace(y.min(), y.max(), resolution)):
                grid[i, j] = self.get_gev_params(np.array([xi, yj])).to_array()
        return grid
