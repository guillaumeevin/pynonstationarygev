import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class AbstractSpatioTemporalCoordinates(AbstractCoordinates):

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None):
        assert cls.COORDINATE_T in df.columns
        assert cls.COORDINATE_X in df.columns
        # Assert that the time steps are in the good order with respect to the coordinates
        nb_points = len(set(df[cls.COORDINATE_X]))
        first_time_step_for_all_points = df.iloc[:nb_points][cls.COORDINATE_T]
        assert len(set(first_time_step_for_all_points)) == 1
        return super().from_df_and_slicer(df, SpatioTemporalSlicer, train_split_ratio)