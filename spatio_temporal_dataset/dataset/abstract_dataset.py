import os
import numpy as np
import os.path as op
import pandas as pd

from spatio_temporal_dataset.dataset.spatio_temporal_split import SpatialTemporalSplit, SpatioTemporalSlicer
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractDataset(object):

    def __init__(self, temporal_observations: AbstractTemporalObservations, coordinates: AbstractCoordinates):
        # is_same_index = temporal_observations.index == coordinates.index  # type: pd.Series
        # assert is_same_index.all()
        self.temporal_observations = temporal_observations
        self.coordinates = coordinates
        self.spatio_temporal_slicer = SpatioTemporalSlicer(coordinate_train_ind=self.coordinates.train_ind,
                                                           observation_train_ind=self.temporal_observations.train_ind)

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        temporal_maxima = AbstractTemporalObservations.from_df(df)
        coordinates = AbstractCoordinates.from_df(df)
        return cls(temporal_maxima, coordinates)

    def to_csv(self, csv_path: str):
        dirname = op.dirname(csv_path)
        if not op.exists(dirname):
            os.makedirs(dirname)
        self.df_dataset.to_csv(csv_path)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        # todo: maybe I should add the split from the temporal observations
        return self.temporal_observations.df_maxima_gev.join(self.coordinates.df_merged)

    def df_coordinates(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all):
        return self.coordinates.df_coordinates(split=split)

    @property
    def coordinates_values(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all):
        return self.coordinates.coordinates_values(split=split)

    def maxima_gev(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all) -> np.ndarray:
        return self.temporal_observations.maxima_gev(split, self.spatio_temporal_slicer)

    def maxima_frech(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all) -> np.ndarray:
        return self.temporal_observations.maxima_frech(split, self.spatio_temporal_slicer)

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: SpatialTemporalSplit = SpatialTemporalSplit.all):
        self.temporal_observations.set_maxima_frech(maxima_frech_values, split, self.spatio_temporal_slicer)