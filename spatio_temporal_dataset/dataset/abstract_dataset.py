import os
import os.path as op
import pandas as pd
from spatio_temporal_dataset.temporal_maxima.temporal_maxima import TemporalMaxima
from spatio_temporal_dataset.spatial_coordinates.abstract_coordinates import AbstractSpatialCoordinates


class AbstractDataset(object):

    def __init__(self, temporal_maxima: TemporalMaxima, spatial_coordinates: AbstractSpatialCoordinates):
        assert (temporal_maxima.index == spatial_coordinates.index).all()
        self.temporal_maxima = temporal_maxima
        self.spatial_coordinates = spatial_coordinates

    @classmethod
    def from_csv(cls, csv_path: str):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        temporal_maxima = TemporalMaxima.from_df(df)
        spatial_coordinates = AbstractSpatialCoordinates.from_df(df)
        return cls(temporal_maxima=temporal_maxima, spatial_coordinates=spatial_coordinates)

    def to_csv(self, csv_path: str):
        dirname = op.dirname(csv_path)
        if not op.exists(dirname):
            os.makedirs(dirname)
        self.df_dataset.to_csv(csv_path)

    @property
    def df_dataset(self) -> pd.DataFrame:
        # Merge dataframes with the maxima and with the coordinates
        return self.temporal_maxima.df_maxima.join(self.spatial_coordinates.df_coord)

    @property
    def coord(self):
        return self.spatial_coordinates.coord

    @property
    def maxima(self):
        return self.temporal_maxima.maxima


class RealDataset(AbstractDataset):
    pass
