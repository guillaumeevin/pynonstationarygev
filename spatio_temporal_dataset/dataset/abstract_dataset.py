import os
import os.path as op
from typing import List

import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.split import Split
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractDataset(object):

    def __init__(self, observations: AbstractSpatioTemporalObservations, coordinates: AbstractCoordinates,
                 slicer_class: type = SpatialSlicer):
        assert pd.Index.equals(observations.index, coordinates.index)
        assert isinstance(slicer_class, type)
        self.observations = observations
        self.coordinates = coordinates
        self.slicer = slicer_class(coordinates_train_ind=self.coordinates.train_ind,
                                   observations_train_ind=self.observations.train_ind)  # type: AbstractSlicer
        assert isinstance(self.slicer, AbstractSlicer)

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        temporal_maxima = AbstractSpatioTemporalObservations.from_df(df)
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
        return self.observations.df_maxima_gev.join(self.coordinates.df_merged)

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        return self.coordinates.df_coordinates(split=split)

    # Observation wrapper

    def maxima_gev(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_gev(split, self.slicer)

    def maxima_frech(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_frech(split, self.slicer)

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all):
        self.observations.set_maxima_frech(maxima_frech_values, split, self.slicer)

    # Coordinates wrapper

    @property
    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.coordinates.coordinates_values(split=split)

    # Slicer wrapper

    @property
    def train_split(self) -> Split:
        return self.slicer.train_split

    @property
    def test_split(self) -> Split:
        return self.slicer.test_split

    @property
    def splits(self) -> List[Split]:
        return self.slicer.splits
