import pandas as pd

from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates


class UniformSpatioTemporalCoordinates(AbstractSpatioTemporalCoordinates):

    @classmethod
    def from_nb_points_and_nb_steps(cls, nb_points, nb_steps, train_split_ratio: float = None):
        assert isinstance(nb_steps, int) and nb_steps >= 1
        df_spatial = UniformSpatialCoordinates.df_spatial(nb_points=nb_points)
        # df_temporal = ConsecutiveTemporalCoordinates.df_temporal(nb_temporal_steps=nb_temporal_steps)
        df_time_steps = []
        for t in range(nb_steps):
            df_time_step = df_spatial.copy()
            df_time_step[cls.COORDINATE_T] = t
            df_time_steps.append(df_time_step)
        df_time_steps = pd.concat(df_time_steps, ignore_index=True)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio)


