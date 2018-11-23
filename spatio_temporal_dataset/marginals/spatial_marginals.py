from spatio_temporal_dataset.marginals.abstract_marginals import AbstractMarginals

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractSpatialCoordinates


class SpatialMarginal(AbstractMarginals):
    """The main idea is to have on marginal per station"""

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates):
        self.spatial_coordinates = spatial_coordinates


class SimulatedSpatialMarginal(SpatialMarginal):
    pass
