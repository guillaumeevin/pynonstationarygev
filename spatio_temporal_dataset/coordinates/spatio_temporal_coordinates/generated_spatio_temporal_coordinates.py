import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class UniformSpatioTemporalCoordinates(AbstractSpatioTemporalCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, nb_time_steps=1, max_radius=1.0):
        assert isinstance(nb_time_steps, int) and nb_time_steps >= 1
        df_spatial = UniformSpatialCoordinates.df_spatial(nb_points)
        df_time_steps = []
        for t in range(nb_time_steps):
            df_time_step = df_spatial.copy()
            df_time_step[cls.COORDINATE_T] = t
            df_time_steps.append(df_time_step)
        df_time_steps = pd.concat(df_time_steps, ignore_index=True)
        print(df_time_steps)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio)


