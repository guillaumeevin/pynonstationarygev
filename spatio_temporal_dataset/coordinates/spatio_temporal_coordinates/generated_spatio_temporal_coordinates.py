import pandas as pd

from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_2D import LinSpaceSpatial2DCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates


class GeneratedSpatioTemporalCoordinates(AbstractSpatioTemporalCoordinates):
    SPATIAL_COORDINATES_CLASS = None

    @classmethod
    def from_nb_points_and_nb_steps(cls, nb_points, nb_steps):
        assert isinstance(nb_steps, int) and nb_steps >= 1
        assert hasattr(cls.spatial_coordinate_class(), 'df_spatial')
        df_spatial = cls.spatial_coordinate_class().df_spatial(nb_points=nb_points)
        return cls.from_df_spatial_and_nb_steps(df_spatial, nb_steps)

    @classmethod
    def spatial_coordinate_class(cls):
        raise NotImplementedError


class UniformSpatioTemporalCoordinates(GeneratedSpatioTemporalCoordinates):

    @classmethod
    def spatial_coordinate_class(cls):
        return UniformSpatialCoordinates


class LinSpaceSpatial2DSpatioTemporalCoordinates(GeneratedSpatioTemporalCoordinates):

    @classmethod
    def spatial_coordinate_class(cls):
        return LinSpaceSpatial2DCoordinates
