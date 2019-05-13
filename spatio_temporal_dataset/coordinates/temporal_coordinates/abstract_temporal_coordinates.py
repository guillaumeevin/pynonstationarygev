import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer


class AbstractTemporalCoordinates(AbstractCoordinates):

    @property
    def temporal_coordinates(self):
        return self

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None, transformation_class: type = None):
        assert cls.COORDINATE_T in df.columns
        assert not any([coordinate_name in df.columns for coordinate_name in cls.COORDINATE_SPATIAL_NAMES])
        return super().from_df_and_slicer(df, TemporalSlicer, train_split_ratio, transformation_class)
