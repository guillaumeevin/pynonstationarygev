import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class AbstractSpatioTemporalCoordinates(AbstractCoordinates):

    def __init__(self, df: pd.DataFrame, slicer_class: type, s_split_spatial: pd.Series = None,
                 s_split_temporal: pd.Series = None):
        super().__init__(df, slicer_class, s_split_spatial, s_split_temporal)
        self.spatial_coordinates = AbstractSpatialCoordinates.from_df(df=self.df_spatial_coordinates())

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
        for t in range(nb_steps):
            df_time_step = df_spatial.copy()
            df_time_step[cls.COORDINATE_T] = start + t
            df_time_step.index = get_index_with_spatio_temporal_index_suffix(df_spatial, t)
            df_time_steps.append(df_time_step)
        df_time_steps = pd.concat(df_time_steps)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio)


