import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.multiple_transformation import \
    MultipleTransformation
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class AbstractSpatioTemporalCoordinates(AbstractCoordinates):

    def __init__(self, df: pd.DataFrame, slicer_class: type,
                 s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None,
                 transformation_class: type = None,
                 spatial_coordinates: AbstractSpatialCoordinates = None,
                 temporal_coordinates: AbstractTemporalCoordinates = None):
        super().__init__(df, slicer_class, s_split_spatial, s_split_temporal, None)
        # Spatial coordinates'
        if spatial_coordinates is None:
            self._spatial_coordinates = AbstractSpatialCoordinates.from_df(
                df=self.df_spatial_coordinates(transformed=False),
                transformation_class=transformation_class)
        else:
            self._spatial_coordinates = spatial_coordinates
        # Temporal coordinates
        if temporal_coordinates is None:
            self._temporal_coordinates = AbstractTemporalCoordinates.from_df(
                df=self.df_temporal_coordinates(transformed=False),
                transformation_class=transformation_class)
        else:
            self._temporal_coordinates = temporal_coordinates
        # Combined the transformations
        self.transformation_class = None
        self.transformation = MultipleTransformation(transformation_1=self.spatial_coordinates.transformation,
                                                     transformation_2=self.temporal_coordinates.transformation)

    @property
    def spatial_coordinates(self):
        return self._spatial_coordinates

    @property
    def temporal_coordinates(self):
        return self._temporal_coordinates

    @classmethod
    def get_df_from_df_spatial_and_coordinate_t_values(cls, coordinate_t_values, df_spatial):
        df_time_steps = []
        for t, coordinate_t_value in enumerate(coordinate_t_values):
            df_time_step = df_spatial.copy()
            df_time_step[cls.COORDINATE_T] = coordinate_t_value
            df_time_step.index = get_index_with_spatio_temporal_index_suffix(df_spatial, t)
            df_time_steps.append(df_time_step)
        df_time_steps = pd.concat(df_time_steps)
        return df_time_steps

    @classmethod
    def from_spatial_coordinates_and_temporal_coordinates(cls, spatial_coordinates: AbstractSpatialCoordinates,
                                                          temporal_coordinates: AbstractTemporalCoordinates):
        df_spatial = spatial_coordinates.df_spatial_coordinates(transformed=False)
        coordinate_t_values = temporal_coordinates.df_temporal_coordinates(transformed=False).iloc[:, 0].values
        df = cls.get_df_from_df_spatial_and_coordinate_t_values(df_spatial=df_spatial,
                                                                coordinate_t_values=coordinate_t_values)
        return cls(df=df, slicer_class=SpatioTemporalSlicer,
                   spatial_coordinates=spatial_coordinates, temporal_coordinates=temporal_coordinates)

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None, transformation_class: type = None):
        assert cls.COORDINATE_T in df.columns
        assert cls.COORDINATE_X in df.columns
        # Assert that the time steps are in the good order with respect to the coordinates
        nb_points = len(set(df[cls.COORDINATE_X]))
        first_time_step_for_all_points = df.iloc[:nb_points][cls.COORDINATE_T]
        assert len(set(first_time_step_for_all_points)) == 1
        return super().from_df_and_slicer(df, SpatioTemporalSlicer, train_split_ratio, transformation_class)

    @classmethod
    def from_df_spatial_and_coordinate_t_values(cls, df_spatial, coordinate_t_values, train_split_ratio: float = None,
                                                transformation_class: type = None):
        df_time_steps = cls.get_df_from_df_spatial_and_coordinate_t_values(coordinate_t_values, df_spatial)
        return cls.from_df(df=df_time_steps, train_split_ratio=train_split_ratio,
                           transformation_class=transformation_class)

    @classmethod
    def from_df_spatial_and_nb_steps(cls, df_spatial, nb_steps, train_split_ratio: float = None, start=0,
                                     transformation_class: type = None):
        coordinate_t_values = [start + t for t in range(nb_steps)]
        return cls.from_df_spatial_and_coordinate_t_values(df_spatial, coordinate_t_values, train_split_ratio,
                                                           transformation_class)

    @classmethod
    def from_df_spatial_and_df_temporal(cls, df_spatial, df_temporal, train_split_ratio: float = None,
                                        transformation_class: type = None):
        nb_steps = len(df_temporal)
        coordinate_t_values = [df_temporal.iloc[t].values[0] for t in range(nb_steps)]
        return cls.from_df_spatial_and_coordinate_t_values(df_spatial, coordinate_t_values, train_split_ratio,
                                                           transformation_class)
