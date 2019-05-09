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

    @classmethod
    def from_df_spatial_and_nb_steps(cls, df_spatial, nb_steps, train_split_ratio: float = None, start=0):
        df_time_steps = []
        index_type = type(df_spatial.index[0])
        for t in range(nb_steps):
            df_time_step = df_spatial.copy()
            df_time_step[cls.COORDINATE_T] = start + t
            index_suffix = index_type(t * len(df_spatial))
            time_step_index = [i + index_suffix for i in df_spatial.index]
            df_time_step.index = time_step_index
            df_time_steps.append(df_time_step)
        df_time_steps = pd.concat(df_time_steps)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio)
