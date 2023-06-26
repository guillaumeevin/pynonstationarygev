import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.utils import get_index_with_spatio_temporal_index_suffix


class AbstractSpatioTemporalCoordinates(AbstractCoordinates):

    def __init__(self, df: pd.DataFrame = None,
                 spatial_coordinates: AbstractSpatialCoordinates = None,
                 temporal_coordinates: AbstractTemporalCoordinates = None):
        df = self.load_df_is_needed(df, spatial_coordinates, temporal_coordinates)
        super().__init__(df)
        # Spatial coordinates'
        if spatial_coordinates is None:
            self._spatial_coordinates = AbstractSpatialCoordinates.from_df(
                df=self.df_spatial_coordinates())
        else:
            self._spatial_coordinates = spatial_coordinates
        # Temporal coordinates
        if temporal_coordinates is None:
            self._temporal_coordinates = AbstractTemporalCoordinates.from_df(
                df=self.df_temporal_coordinates())
        else:
            self._temporal_coordinates = temporal_coordinates

    def load_df_is_needed(self, df, spatial_coordinates, temporal_coordinates):
        if df is None:
            assert spatial_coordinates is not None and temporal_coordinates is not None
            df = self.get_df_from_spatial_and_temporal_coordinates(spatial_coordinates, temporal_coordinates)
        return df

    @property
    def spatial_coordinates(self):
        return self._spatial_coordinates

    @property
    def temporal_coordinates(self):
        return self._temporal_coordinates

    @classmethod
    def from_spatial_coordinates_and_temporal_coordinates(cls, spatial_coordinates: AbstractSpatialCoordinates,
                                                          temporal_coordinates: AbstractTemporalCoordinates):
        df = cls.get_df_from_spatial_and_temporal_coordinates(spatial_coordinates, temporal_coordinates)
        return cls(df=df, spatial_coordinates=spatial_coordinates, temporal_coordinates=temporal_coordinates)

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
    def get_df_from_spatial_and_temporal_coordinates(cls, spatial_coordinates, temporal_coordinates):
        # Transformed is False, so that the value in the spatio temporal datasets are still readable
        df_spatial = spatial_coordinates.df_spatial_coordinates()
        coordinate_t_values = temporal_coordinates.df_temporal_coordinates().iloc[:, 0].values
        df = cls.get_df_from_df_spatial_and_coordinate_t_values(df_spatial=df_spatial,
                                                                coordinate_t_values=coordinate_t_values)
        return df

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        assert cls.COORDINATE_T in df.columns
        assert cls.COORDINATE_X in df.columns
        # Assert that the time steps are in the good order with respect to the coordinates
        nb_points = len(set(df[cls.COORDINATE_X]))
        first_time_step_for_all_points = df.iloc[:nb_points][cls.COORDINATE_T]
        assert len(set(first_time_step_for_all_points)) == 1
        return super().from_df(df)

    @classmethod
    def from_df_spatial_and_coordinate_t_values(cls, df_spatial, coordinate_t_values):
        df_time_steps = cls.get_df_from_df_spatial_and_coordinate_t_values(coordinate_t_values, df_spatial)
        return cls.from_df(df=df_time_steps)

    @classmethod
    def from_df_spatial_and_nb_steps(cls, df_spatial, nb_steps, start=0):
        coordinate_t_values = [start + t for t in range(nb_steps)]
        return cls.from_df_spatial_and_coordinate_t_values(df_spatial, coordinate_t_values)

    @classmethod
    def from_df_spatial_and_df_temporal(cls, df_spatial, df_temporal):
        nb_steps = len(df_temporal)
        coordinate_t_values = [df_temporal.iloc[t].values[0] for t in range(nb_steps)]
        return cls.from_df_spatial_and_coordinate_t_values(df_spatial, coordinate_t_values)
