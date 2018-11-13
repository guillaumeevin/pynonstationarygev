import os
import numpy as np
import os.path as op
import pandas as pd
from spatio_temporal_dataset.temporal_observations.abstract_temporal_observations import AbstractTemporalObservations
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class AbstractDataset(object):

    def __init__(self, temporal_observations: AbstractTemporalObservations, spatial_coordinates: AbstractSpatialCoordinates):
        # assert
        # is_same_index = temporal_observations.index == spatial_coordinates.index  # type: pd.Series
        # assert is_same_index.all()
        self.temporal_observations = temporal_observations
        self.spatial_coordinates = spatial_coordinates

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        temporal_maxima = AbstractTemporalObservations.from_df(df)
        spatial_coordinates = AbstractSpatialCoordinates.from_df(df)
        return cls(temporal_maxima, spatial_coordinates)

    def to_csv(self, csv_path: str):
        dirname = op.dirname(csv_path)
        if not op.exists(dirname):
            os.makedirs(dirname)
        self.df_dataset.to_csv(csv_path)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.temporal_observations.df_maxima.join(self.spatial_coordinates.df_coord)

    @property
    def coord(self):
        return self.spatial_coordinates.coord

    @property
    def maxima(self) -> np.ndarray:
        return self.temporal_observations.maxima

    @property
    def maxima_normalized(self):
        return self.temporal_observations.maxima_normalized

    @maxima_normalized.setter
    def maxima_normalized(self, maxima_normalized_to_set):
        self.temporal_observations.maxima_normalized = maxima_normalized_to_set

