import copy
import os
import os.path as op
from typing import List, Dict

import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class AbstractDataset(object):

    def __init__(self, observations: AbstractSpatioTemporalObservations, coordinates: AbstractCoordinates):
        assert pd.Index.equals(observations.index, coordinates.index), '\n{}\n{}'.format(observations.index, coordinates.index)
        self.observations = observations  # type: AbstractSpatioTemporalObservations
        self.coordinates = coordinates  # type: AbstractCoordinates
        self.subset_id_to_column_idxs = None  # type: Dict[int, List[int]]

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path, index_col=0)
        coordinates = AbstractCoordinates.from_df(df)
        temporal_maxima = AbstractSpatioTemporalObservations.from_df(df)
        return cls(temporal_maxima, coordinates)

    def to_csv(self, csv_path: str):
        dirname = op.dirname(csv_path)
        if not op.exists(dirname):
            os.makedirs(dirname)
        self.df_dataset.to_csv(csv_path)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.observations.df_maxima_merged.join(self.coordinates.df_merged)

    # Observation wrapper

    def maxima_gev(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_gev(split, self.slicer)

    def maxima_frech(self, split: Split = Split.all) -> np.ndarray:
        return self.observations.maxima_frech(split, self.slicer)

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all):
        self.observations.set_maxima_frech(maxima_frech_values, split, self.slicer)

    # Coordinates wrapper

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        return self.coordinates.df_coordinates(split=split)

    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.coordinates.coordinates_values(split=split)

    def coordinates_index(self, split: Split = Split.all) -> pd.Index:
        return self.coordinates.coordinates_index(split=split)

    # Slicer wrapper

    @property
    def slicer(self) -> AbstractSlicer:
        return self.coordinates.slicer

    @property
    def train_split(self) -> Split:
        return self.slicer.train_split

    @property
    def test_split(self) -> Split:
        return self.slicer.test_split

    @property
    def splits(self) -> List[Split]:
        return self.slicer.splits

    # Dataset subsets

    def create_subsets(self, nb_subsets):
        self.subset_id_to_column_idxs = {}
        for subset_id in range(nb_subsets):
            column_idxs = [idx for idx in range(self.observations.nb_obs) if idx % nb_subsets == subset_id]
            self.subset_id_to_column_idxs[subset_id] = column_idxs


def get_subset_dataset(dataset: AbstractDataset, subset_id) -> AbstractDataset:
    columns_idxs = dataset.subset_id_to_column_idxs[subset_id]
    assert dataset.subset_id_to_column_idxs is not None, 'You need to create subsets'
    assert subset_id in dataset.subset_id_to_column_idxs.keys()
    subset_dataset = copy.deepcopy(dataset)
    observations = subset_dataset.observations
    if observations.df_maxima_gev is not None:
        observations.df_maxima_gev = observations.df_maxima_gev.iloc[:, columns_idxs]
    if observations.df_maxima_frech is not None:
        observations.df_maxima_frech = observations.df_maxima_frech.iloc[:, columns_idxs]
    return subset_dataset
