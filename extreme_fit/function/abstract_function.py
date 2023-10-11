import numpy as np

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractFunction(object):

    def __init__(self, coordinates: AbstractCoordinates):
        self.coordinates = coordinates

