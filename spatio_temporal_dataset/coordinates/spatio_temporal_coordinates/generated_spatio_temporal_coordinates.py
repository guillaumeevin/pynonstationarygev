import pandas as pd

from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates


class GeneratedSpatioTemporalCoordinates(AbstractSpatioTemporalCoordinates):
    SPATIAL_COORDINATES_CLASS = None

    @classmethod
    def from_nb_points_and_nb_steps(cls, nb_points, nb_steps, train_split_ratio: float = None):
        assert isinstance(nb_steps, int) and nb_steps >= 1
        assert cls.SPATIAL_COORDINATES_CLASS is not None
        assert hasattr(cls.SPATIAL_COORDINATES_CLASS, 'df_spatial')
        df_spatial = cls.SPATIAL_COORDINATES_CLASS.df_spatial(nb_points=nb_points)
        df_time_steps = cls.generate_df_spatio_temporal(df_spatial, nb_steps)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio)


class UniformSpatioTemporalCoordinates(GeneratedSpatioTemporalCoordinates):
    SPATIAL_COORDINATES_CLASS = UniformSpatialCoordinates


class LinSpaceSpatial2DSpatioTemporalCoordinates(GeneratedSpatioTemporalCoordinates):
    SPATIAL_COORDINATES_CLASS = LinSpaceSpatial2DCoordinates
