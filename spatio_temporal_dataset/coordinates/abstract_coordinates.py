import os.path as op
from itertools import chain
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    AbstractTransformation, IdentityTransformation
from spatio_temporal_dataset.coordinates.utils import get_index_without_spatio_temporal_index_suffix


class AbstractCoordinates(object):
    """
    Main attribute of the class is the DataFrame df_all_coordinates
    Index are coordinates index
    Columns are the value of each coordinates

    """
    # Spatial columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_SPATIAL_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    SPATIAL_SPLIT = 'spatial_split'
    # Temporal columns
    COORDINATE_T = 'coord_t'
    # Climate model columns
    COORDINATE_RCP = 'coord_rcp'
    COORDINATE_GCM = 'coord_gcm'
    COORDINATE_RCM = 'coord_rcm'
    COORDINATE_CLIMATE_MODEL_NAMES = [COORDINATE_RCP, COORDINATE_GCM, COORDINATE_RCM]
    # Coordinates columns
    COORDINATES_NAMES = COORDINATE_SPATIAL_NAMES + [COORDINATE_T] + COORDINATE_CLIMATE_MODEL_NAMES
    # Coordinate type
    ALL_COORDINATES_ACCEPTED_TYPES = ['int64', 'float64']
    COORDINATE_TYPE = 'float64'

    def __init__(self, df: pd.DataFrame, transformation_class: type = None):
        # Extract df_all_coordinates from df
        coordinate_columns = [c for c in df.columns if c in self.COORDINATES_NAMES]
        assert len(coordinate_columns) > 0
        # Sort coordinates according to a specified order
        sorted_coordinates_columns = [c for c in self.COORDINATES_NAMES if c in coordinate_columns]
        self.df_all_coordinates = df.loc[:, sorted_coordinates_columns].copy()  # type: pd.DataFrame
        # Cast coordinates
        ind = self.df_all_coordinates.columns.isin(self.COORDINATE_CLIMATE_MODEL_NAMES)
        self.df_coordinate_climate_model = self.df_all_coordinates.loc[:, ind].copy()
        self.df_all_coordinates = self.df_all_coordinates.loc[:, ~ind] # type: pd.DataFrame
        self.df_all_coordinates = self.df_all_coordinates.astype(self.COORDINATE_TYPE)  # type: pd.DataFrame

        # Transformation attribute
        if transformation_class is None:
            transformation_class = IdentityTransformation
        self.transformation_class = transformation_class  # type: type
        # Transformation only works for float coordinates
        accepted_dtypes = [self.COORDINATE_TYPE]
        assert len(self.df_all_coordinates.select_dtypes(include=accepted_dtypes).columns) \
               == len(coordinate_columns) - sum(ind), 'coordinates columns dtypes should belong to {}'.format(accepted_dtypes)
        # Transformation class is instantiated with all coordinates
        self.transformation = transformation_class(self.df_all_coordinates)  # type: AbstractTransformation
        assert isinstance(self.transformation, AbstractTransformation)
        # Some parameters to set if needed
        self.gcm_rcm_couple_as_pseudo_truth = None

    # ClassMethod constructor

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        return cls(df=df)

    @classmethod
    def from_df_and_transformation_class(cls, df: pd.DataFrame, transformation_class: type = None):
        # All the index should be unique
        assert len(set(df.index)) == len(df), 'df indices are not unique'
        return cls(df=df, transformation_class=transformation_class)

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        # Index correspond to the first column
        index_column_name = df.columns[0]
        assert index_column_name not in cls.COORDINATE_SPATIAL_NAMES
        df.set_index(index_column_name, inplace=True)
        return cls.from_df(df)

    @property
    def index(self) -> pd.Index:
        return self.df_all_coordinates.index

    # Split

    def df_coordinates(self, transformed=True, add_climate_informations=False) -> pd.DataFrame:
        if transformed:
            df_transformed_coordinates = self.transformation.transform_df(self.df_all_coordinates)
        else:
            df_transformed_coordinates = self.df_all_coordinates
        if add_climate_informations:
            df_transformed_coordinates = pd.concat([df_transformed_coordinates,
                                                    self.df_coordinate_climate_model], axis=1)
        return df_transformed_coordinates

    def coordinates_values(self, transformed=True) -> np.ndarray:
        return self.df_coordinates(transformed=transformed).values

    @property
    def coordinates_names(self) -> List[str]:
        return self.spatial_coordinates_names + self.temporal_coordinates_names

    @property
    def nb_coordinates(self) -> int:
        return len(self.coordinates_names)

    @property
    def coordinates_dims(self) -> List[int]:
        return list(range(self.nb_coordinates))

    # Spatial attributes

    @property
    def spatial_coordinates_dims(self):
        return list(range(self.nb_spatial_coordinates))

    @property
    def spatial_coordinates_names(self) -> List[str]:
        return [name for name in self.COORDINATE_SPATIAL_NAMES if name in self.df_all_coordinates.columns]

    @property
    def nb_spatial_coordinates(self) -> int:
        return len(self.spatial_coordinates_names)

    @property
    def has_spatial_coordinates(self) -> bool:
        return self.nb_spatial_coordinates > 0

    def df_spatial_coordinates(self, transformed=True, drop_duplicates=True) -> pd.DataFrame:
        if self.nb_spatial_coordinates == 0:
            return pd.DataFrame()
        else:
            df = self.df_coordinates(transformed).loc[:, self.spatial_coordinates_names]
            return df.drop_duplicates() if drop_duplicates else df

    @property
    def spatial_index(self) -> pd.Index:
        df_spatial = self.df_spatial_coordinates()
        if self.has_spatio_temporal_coordinates:
            # Remove the spatio temporal index suffix
            return get_index_without_spatio_temporal_index_suffix(df_spatial)
        else:
            return df_spatial.index

    # Temporal attributes

    @property
    def temporal_coordinates_names(self) -> List[str]:
        return [self.COORDINATE_T] if self.COORDINATE_T in self.df_all_coordinates else []

    @property
    def nb_temporal_coordinates(self) -> int:
        return len(self.temporal_coordinates_names)

    @property
    def has_temporal_coordinates(self) -> bool:
        return self.nb_temporal_coordinates > 0

    def df_temporal_coordinates(self, transformed=True, drop_duplicates=True) -> pd.DataFrame:
        if self.nb_temporal_coordinates == 0:
            return pd.DataFrame()
        else:
            df = self.df_coordinates(transformed=transformed).loc[:, self.temporal_coordinates_names]
            if drop_duplicates:
                return df.drop_duplicates()
            else:
                return df

    def df_temporal_coordinates_for_fit(self, starting_point=None,
                                        temporal_covariate_for_fit: Union[None, type] = None,
                                        drop_duplicates=True, climate_coordinates_with_effects=None) -> pd.DataFrame:
        # Load time covariate
        if starting_point is None:
            df = self.df_temporal_coordinates(transformed=True, drop_duplicates=drop_duplicates)
        else:
            # Load the un transformed coordinates
            df_temporal_coordinates = self.df_temporal_coordinates(transformed=False)
            # If starting point is not None, the transformation has not yet been applied
            # thus we need to modify the coordinate with the starting point, and then to apply the transformation
            # Compute the indices to modify
            ind_to_modify = df_temporal_coordinates.iloc[:, 0] <= starting_point  # type: pd.Series
            # Assert that some coordinates are selected but not all
            msg = '{} First year={} Last_year={}'.format(sum(ind_to_modify), df_temporal_coordinates.iloc[0, 0],
                                                         df_temporal_coordinates.iloc[-1, 0])
            assert 0 < sum(ind_to_modify) < len(ind_to_modify), msg
            # Modify the temporal coordinates to enforce the stationarity
            df_temporal_coordinates.loc[ind_to_modify] = starting_point
            # Load the temporal transformation object
            temporal_transformation = self.temporal_coordinates.transformation_class(df_temporal_coordinates)  # type: AbstractTransformation
            # Return the result of the temporal transformation
            df = temporal_transformation.transform_df(df_temporal_coordinates)

        # Potentially transform the time covariate into another covariate
        if temporal_covariate_for_fit is not None:
            df_input = pd.concat([df, self.df_coordinate_climate_model], axis=1)
            df.loc[:, self.COORDINATE_T] = df_input.apply(temporal_covariate_for_fit.get_temporal_covariate, axis=1)
        if climate_coordinates_with_effects is not None:
            assert all([c in AbstractCoordinates.COORDINATE_CLIMATE_MODEL_NAMES for c in climate_coordinates_with_effects])
            for climate_coordinate in climate_coordinates_with_effects:
                assert climate_coordinate in AbstractCoordinates.COORDINATE_CLIMATE_MODEL_NAMES

                df_coordinate_climate_model = self.df_coordinate_climate_model.copy()
                # Potentially remove the climate coordinates for some gcm rcm couple
                # We cannot do it sooner because we need the name of the GCM and RCM to find the appropriate temperature
                if self.gcm_rcm_couple_as_pseudo_truth is not None:
                    gcm, rcm = self.gcm_rcm_couple_as_pseudo_truth
                    ind = (df_coordinate_climate_model[self.COORDINATE_GCM] == gcm) \
                          & (df_coordinate_climate_model[self.COORDINATE_RCM] == rcm)
                    df_coordinate_climate_model.loc[ind, self.COORDINATE_CLIMATE_MODEL_NAMES] = None
                # Create some additional columns
                only_observation_with_empty_climate_model_columns = df_coordinate_climate_model.isnull().all().all()
                if not only_observation_with_empty_climate_model_columns:
                    s, has_observations, unique_values_without_nan = self.load_unique_values(climate_coordinate, df_coordinate_climate_model)
                    if has_observations:
                        for value_name in unique_values_without_nan:
                            serie_is_value = (s == value_name) * 1
                            df[value_name] = serie_is_value
                    else:
                        raise NotImplementedError
                        # todo: the coordinate for three gcm should be 1, 0 then 0, 1 finally -1 -1
                        # maybe it not exactly that, but in this case (without observaitons),
                        # i need to ensure a constraint that the sum of coef is zero

        return df

    def load_unique_values(self, climate_coordinate, df_coordinate_climate_model=None):
        if df_coordinate_climate_model is None:
            df_coordinate_climate_model = self.df_coordinate_climate_model
        s = df_coordinate_climate_model[climate_coordinate]
        for character in self.character_to_remove_from_climate_model_coordinate_name():
            s = s.str.replace(character, "")
        unique_values = s.unique()
        unique_values_without_nan = [v for v in unique_values if isinstance(v, str)]
        has_observations = len(unique_values) == len(unique_values_without_nan) + 1
        # Remove if need the names of the pseudo ground truth
        if self.gcm_rcm_couple_as_pseudo_truth is not None:
            gcm_rcm_couple_names_for_fit = [self.climate_model_coordinate_name_to_name_for_fit(name)
                                            for name in self.gcm_rcm_couple_as_pseudo_truth]
            unique_values_without_nan = [name for name in unique_values_without_nan if name not in gcm_rcm_couple_names_for_fit]
        return s, has_observations, unique_values_without_nan

    def load_ordered_columns_names(self, climate_coordinates_names_with_effects):
        column_names = []
        for climate_coordinate in climate_coordinates_names_with_effects:
            _, _, names = self.load_unique_values(climate_coordinate)
            column_names.extend(names)
        return column_names

    @classmethod
    def load_full_climate_coordinates_with_effects(cls, param_name_to_climate_coordinates_with_effects):
        two_climate_coordinates_considered = [cls.COORDINATE_GCM, cls.COORDINATE_RCM]
        all_climate_coordinate_with_effects = set(chain(*[c
                                                          for c
                                                          in param_name_to_climate_coordinates_with_effects.values()
                                                          if c is not None]))
        assert all([c in two_climate_coordinates_considered for c in all_climate_coordinate_with_effects])
        if len(all_climate_coordinate_with_effects) == 2:
            return two_climate_coordinates_considered
        elif len(all_climate_coordinate_with_effects) == 1:
            return list(all_climate_coordinate_with_effects)
        else:
            return None

    def df_climate_models(self):
        return self.df_coordinate_climate_model

    @classmethod
    def character_to_remove_from_climate_model_coordinate_name(cls):
        return ['-']

    @classmethod
    def climate_model_coordinate_name_to_name_for_fit(cls, name):
        for c in cls.character_to_remove_from_climate_model_coordinate_name():
            return name.replace(c, "")

    @property
    def temporal_coordinates(self):
        raise NotImplementedError

    @property
    def nb_steps(self) -> int:
        return len(self.df_temporal_coordinates())

    def df_temporal_range(self) -> Tuple[int, int]:
        df_temporal_coordinates = self.df_temporal_coordinates()
        return int(df_temporal_coordinates.min()), int(df_temporal_coordinates.max()),

    @property
    def idx_temporal_coordinates(self):
        return self.coordinates_names.index(self.COORDINATE_T)

    @property
    def idx_x_coordinates(self):
        return self.coordinates_names.index(self.COORDINATE_X)

# Spatio temporal attributes

    @property
    def has_spatio_temporal_coordinates(self) -> bool:
        return self.has_spatial_coordinates and self.has_temporal_coordinates

    def spatio_temporal_shape(self) -> Tuple[int, int]:
        return len(self.df_spatial_coordinates()), len(self.df_temporal_coordinates())

    def ind_of_df_all_coordinates(self, coordinate_name, value):
        return self.df_all_coordinates.loc[:, coordinate_name] == value

    @property
    def coordinate_name_to_dim(self):
        return {v: k for k, v in self.dim_to_coordinate.items()}

    @property
    def dim_to_coordinate(self):
        return dict(enumerate(self.coordinates_names))

    #  Visualization

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_X].values.copy()

    @property
    def y_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_Y].values.copy()

    @property
    def z_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_Z].values.copy()

    @property
    def t_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_T].values.copy()

    def visualize(self):
        if self.nb_spatial_coordinates == 1:
            self.visualization_1D()
        elif self.nb_spatial_coordinates == 2:
            self.visualization_2D()
        else:
            self.visualization_3D()

    def visualization_1D(self):
        assert self.nb_spatial_coordinates >= 1
        x = self.x_coordinates
        y = np.zeros(len(x))
        plt.scatter(x, y)
        plt.show()

    def visualization_2D(self):
        assert self.nb_spatial_coordinates >= 2
        plt.scatter(self.x_coordinates, self.y_coordinates)
        plt.show()

    def visualization_3D(self):
        assert self.nb_spatial_coordinates == 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
        ax.scatter(self.x_coordinates, self.y_coordinates, self.z_coordinates, marker='^')
        plt.show()

    #  Magic Methods

    def __len__(self):
        return len(self.df_all_coordinates)

    def __mul__(self, other: float):
        self.df_all_coordinates *= other
        return self

    def __rmul__(self, other):
        return self * other

    def __str__(self):
        return pd.concat([self.df_coordinates(), self.df_coordinate_climate_model], axis=1).__str__()
