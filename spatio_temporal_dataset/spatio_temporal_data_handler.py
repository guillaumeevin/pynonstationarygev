from typing import List
import pandas as pd
from spatio_temporal_dataset.marginals.abstract_marginals import AbstractMarginals
from spatio_temporal_dataset.marginals.spatial_marginals import SpatialMarginal
from spatio_temporal_dataset.stations.station import Station, load_stations_from_dataframe
from spatio_temporal_dataset.stations.station_distance import EuclideanDistance2D, StationDistance
import pickle
import os.path as op
from itertools import combinations


class SpatioTemporalDataHandler(object):

    def __init__(self, marginals_class: type, stations: List[Station], station_distance: StationDistance):
        self.stations = stations

        #  Compute once the distances between stations
        for station1, station2 in combinations(self.stations, 2):
            distance = station_distance.compute_distance(station1, station2)
            station1.distance[station2] = distance
            station2.distance[station1] = distance

        # Compute the marginals
        self.marginals = marginals_class(self.stations)  # type: AbstractMarginals

        # Define the max stable
        # self.max_stable =

        print(self.marginals.gev_parameters)

    @classmethod
    def from_dataframe(cls, df):
        return cls.from_spatial_dataframe(df)

    @classmethod
    def from_spatial_dataframe(cls, df):
        stations = load_stations_from_dataframe(df)
        marginal_class = SpatialMarginal
        station_distance = EuclideanDistance2D()
        return cls(marginals_class=marginal_class, stations=stations, station_distance=station_distance)


def get_spatio_temporal_data_handler(pickle_path: str, load_pickle: bool = True, dump_pickle: bool = False, *args) \
        -> SpatioTemporalDataHandler:
    # Either load or dump pickle of a SpatioTemporalDataHandler object
    assert load_pickle or dump_pickle
    if load_pickle:
        assert op.exists(pickle_path) and not dump_pickle
        spatio_temporal_experiment = pickle.load(pickle_path)
    else:
        assert not op.exists(pickle_path)
        spatio_temporal_experiment = SpatioTemporalDataHandler(*args)
        pickle.dump(spatio_temporal_experiment, file=pickle_path)
    return spatio_temporal_experiment


if __name__ == '__main__':
    df = pd.DataFrame(1, index=['station1', 'station2'], columns=['200' + str(i) for i in range(18)])
    xp = SpatioTemporalDataHandler.from_dataframe(df)
