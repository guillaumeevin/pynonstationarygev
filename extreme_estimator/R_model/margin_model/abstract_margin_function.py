from typing import List

import numpy as np

from extreme_estimator.R_model.gev.gev_parameters import GevParams
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class AbstractMarginFunction(object):
    """
    It represents any function mapping points from a space S (could be 2D, 3D,...) to R^3 (the 3 parameters of the GEV)
    """

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams):
        self.spatial_coordinates = spatial_coordinates
        self.default_params = default_params

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        pass


class ConstantMarginFunction(AbstractMarginFunction):

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        return self.default_params


class LinearMarginFunction(AbstractMarginFunction):

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams,
                 linear_dims: List[int]):
        super().__init__(spatial_coordinates, default_params)
        self.linear_dims = linear_dims

# class LinearShapeMarginFunction(AbstractMarginFunction):
#     """Linear function """
#
#     def __init__(self, coordinates, dimension_index_for_linearity=0):
#         super().__init__(coordinates)
#         self.dimension_index_for_linearity = dimension_index_for_linearity
#         assert dimension_index_for_linearity < np.ndim(self.coordinates)
#         # Compute
#
#     def get_gev_params(self, coordinate):
