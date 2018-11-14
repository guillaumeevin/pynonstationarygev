from typing import List, Dict

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

from extreme_estimator.R_model.gev.gev_parameters import GevParams
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class AbstractMarginFunction(object):
    """
    It represents any function mapping points from a space S (could be 2D, 3D,...) to R^3 (the 3 parameters of the GEV)
    """

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams):
        self.spatial_coordinates = spatial_coordinates
        self.default_params = default_params.to_dict()

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        pass

    def visualize_2D(self, gev_param_name=GevParams.GEV_LOC, show=False):
        x = self.spatial_coordinates.x_coordinates
        y = self.spatial_coordinates.y_coordinates
        resolution = 100
        grid = np.zeros([resolution, resolution, 3])
        for i, xi in enumerate(np.linspace(x.min(), x.max(), resolution)):
            for j, yj in enumerate(np.linspace(y.min(), y.max(), resolution)):
                grid[i, j] = self.get_gev_params(np.array([xi, yj])).to_array()
        gev_param_idx = GevParams.GEV_PARAM_NAMES.index(gev_param_name)
        plt.imshow(grid[..., gev_param_idx], extent=(x.min(), x.max(), y.min(), y.max()),
                   interpolation='nearest', cmap=cm.gist_rainbow)
        if show:
            plt.show()
