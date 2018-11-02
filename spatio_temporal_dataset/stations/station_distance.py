from spatio_temporal_dataset.stations.station import Station
import numpy as np


class StationDistance(object):

    @classmethod
    def compute_distance(self, station1: Station, station2: Station) -> float:
        return np.nan


def euclidean_distance(arr1: np.ndarray, arr2: np.ndarray) -> float:
    return np.linalg.norm(arr1 - arr2)


class EuclideanDistance2D(StationDistance):

    @classmethod
    def compute_distance(self, station1: Station, station2: Station) -> float:
        print(station1.latitude)
        stations_coordinates = [np.array([station.latitude, station.longitude]) for station in [station1, station2]]
        return euclidean_distance(*stations_coordinates)
