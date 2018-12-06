import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class AbstractSpatioTemporalCoordinates(AbstractCoordinates):

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None):
        assert cls.COORDINATE_T in df.columns
        assert cls.COORDINATE_X in df.columns
        return super().from_df_and_slicer(df, SpatioTemporalSlicer, train_split_ratio)