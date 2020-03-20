import numpy as np

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractFunction(object):

    def __init__(self, coordinates: AbstractCoordinates):
        self.coordinates = coordinates

    def transform(self, coordinate: np.ndarray) -> np.ndarray:
        return self.coordinates.transformation.transform_array(coordinate)
