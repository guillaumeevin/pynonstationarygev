import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer


class AbstractSpatialCoordinates(AbstractCoordinates):

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None):
        assert cls.COORDINATE_X in df.columns
        assert cls.COORDINATE_T not in df.columns
        return super().from_df_and_slicer(df, SpatialSlicer, train_split_ratio)
