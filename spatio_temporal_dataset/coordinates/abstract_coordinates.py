import os.path as op
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    AbstractTransformation, IdentityTransformation
from spatio_temporal_dataset.coordinates.utils import get_index_without_spatio_temporal_index_suffix
from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer, df_sliced
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from spatio_temporal_dataset.slicer.split import s_split_from_df, ind_train_from_s_split, Split
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer


class AbstractCoordinates(object):
    """
    Main attribute of the class is the DataFrame df_all_coordinates
    Index are coordinates index
    Columns are the value of each coordinates

    So far, the train_split_ratio is the same between the spatial part of the data, and the temporal part
    """
    # Spatial columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_SPATIAL_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    SPATIAL_SPLIT = 'spatial_split'
    # Temporal columns
    COORDINATE_T = 'coord_t'
    TEMPORAL_SPLIT = 'temporal_split'
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

    def __init__(self, df: pd.DataFrame, slicer_class: type, s_split_spatial: pd.Series = None,
                 s_split_temporal: pd.Series = None, transformation_class: type = None):
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
        self.df_all_coordinates = self.df_all_coordinates.astype(self.COORDINATE_TYPE)

        # Slicing attributes
        self.s_split_spatial = s_split_spatial  # type: pd.Series
        self.s_split_temporal = s_split_temporal  # type: pd.Series
        self.slicer = None  # type: Union[None, AbstractSlicer]

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

        # Load the slicer
        if slicer_class is TemporalSlicer:
            self.slicer = TemporalSlicer(self.ind_train_temporal)
        elif slicer_class is SpatialSlicer:
            self.slicer = SpatialSlicer(self.ind_train_spatial)
        elif slicer_class is SpatioTemporalSlicer:
            self.slicer = SpatioTemporalSlicer(self.ind_train_spatial, self.ind_train_temporal)
        else:
            raise ValueError("Unknown slicer_class: {}".format(slicer_class))

    # ClassMethod constructor

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        # Extract the split if they are specified
        s_split_spatial = df[cls.SPATIAL_SPLIT].copy() if cls.SPATIAL_SPLIT in df.columns else None
        s_split_temporal = df[cls.TEMPORAL_SPLIT].copy() if cls.TEMPORAL_SPLIT in df.columns else None

        slicer_class = cls.slicer_class_from_s_splits(s_split_spatial, s_split_temporal)

        return cls(df=df, slicer_class=slicer_class, s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal)

    @classmethod
    def slicer_class_from_s_splits(cls, s_split_spatial, s_split_temporal):
        # Infer the slicer class
        if s_split_temporal is None and s_split_spatial is None:
            raise ValueError('Both split are unspecified')
        elif s_split_temporal is not None and s_split_spatial is None:
            slicer_class = TemporalSlicer
        elif s_split_temporal is None and s_split_spatial is not None:
            slicer_class = SpatialSlicer
        else:
            slicer_class = SpatioTemporalSlicer
        return slicer_class

    @classmethod
    def from_df_and_slicer(cls, df: pd.DataFrame, slicer_class: type, train_split_ratio: float = None,
                           transformation_class: type = None):
        # All the index should be unique
        assert len(set(df.index)) == len(df), 'df indices are not unique'

        # Create a spatial split
        s_split_spatial = cls.spatial_s_split_from_df(df, train_split_ratio)
        # Create a temporal split
        s_split_temporal = cls.temporal_s_split_from_df(df, train_split_ratio)

        return cls(df=df, slicer_class=slicer_class, s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal,
                   transformation_class=transformation_class)

    @classmethod
    def spatial_s_split_from_df(cls, df, train_split_ratio):
        return s_split_from_df(df, cls.COORDINATE_X, cls.SPATIAL_SPLIT, train_split_ratio, True)

    @classmethod
    def temporal_s_split_from_df(cls, df, train_split_ratio):
        return s_split_from_df(df, cls.COORDINATE_T, cls.TEMPORAL_SPLIT, train_split_ratio, False)

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

    @property
    def df_merged(self) -> pd.DataFrame:
        # Merged DataFrame of df_coord with s_split
        return self.df_coordinates().join(self.df_split)

    # Split

    def df_coordinates(self, split: Split = Split.all, transformed=True, add_climate_informations=False) -> pd.DataFrame:
        if transformed:
            df_transformed_coordinates = self.transformation.transform_df(self.df_all_coordinates)
        else:
            df_transformed_coordinates = self.df_all_coordinates
        if add_climate_informations:
            df_transformed_coordinates = pd.concat([df_transformed_coordinates,
                                                    self.df_coordinate_climate_model], axis=1)
        return df_sliced(df=df_transformed_coordinates, split=split, slicer=self.slicer)

    def coordinates_values(self, split: Split = Split.all, transformed=True) -> np.ndarray:
        return self.df_coordinates(split, transformed=transformed).values

    def coordinates_index(self, split: Split = Split.all) -> pd.Index:
        return self.df_coordinates(split).index

    @property
    def ind_train_spatial(self) -> pd.Series:
        return ind_train_from_s_split(s_split=self.s_split_spatial)

    @property
    def ind_train_temporal(self) -> pd.Series:
        return ind_train_from_s_split(s_split=self.s_split_temporal)

    @property
    def df_split(self) -> pd.DataFrame:
        split_name_to_s_split = {
            self.SPATIAL_SPLIT: self.s_split_spatial,
            self.TEMPORAL_SPLIT: self.s_split_temporal,
        }
        # Delete None s_split from the dictionary
        split_name_to_s_split = {k: v for k, v in split_name_to_s_split.items() if v is not None}
        # Create df_split from dict
        return pd.DataFrame.from_dict(split_name_to_s_split)

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

    def df_spatial_coordinates(self, split: Split = Split.all, transformed=True, drop_duplicates=True) -> pd.DataFrame:
        if self.nb_spatial_coordinates == 0:
            return pd.DataFrame()
        else:
            df = self.df_coordinates(split, transformed).loc[:, self.spatial_coordinates_names]
            return df.drop_duplicates() if drop_duplicates else df

    def nb_stations(self, split: Split = Split.all) -> int:
        return len(self.df_spatial_coordinates(split))

    def spatial_index(self, split: Split = Split.all) -> pd.Index:
        df_spatial = self.df_spatial_coordinates(split)
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

    def df_temporal_coordinates(self, split: Split = Split.all, transformed=True,
                                drop_duplicates=True) -> pd.DataFrame:
        if self.nb_temporal_coordinates == 0:
            return pd.DataFrame()
        else:
            df = self.df_coordinates(split, transformed=transformed).loc[:, self.temporal_coordinates_names]
            if drop_duplicates:
                return df.drop_duplicates()
            else:
                return df

    def df_temporal_coordinates_for_fit(self, split=Split.all, starting_point=None,
                                        temporal_covariate_for_fit: Union[None, type] = None,
                                        drop_duplicates=True) -> pd.DataFrame:
        # Load time covariate
        if starting_point is None:
            df = self.df_temporal_coordinates(split=split, transformed=True, drop_duplicates=drop_duplicates)
        else:
            # Load the un transformed coordinates
            df_temporal_coordinates = self.df_temporal_coordinates(split=split, transformed=False)
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
            df_climate_model = df_sliced(df=self.df_coordinate_climate_model, split=split, slicer=self.slicer)
            df_input = pd.concat([df, df_climate_model], axis=1)
            df.loc[:, self.COORDINATE_T] = df_input.apply(temporal_covariate_for_fit.get_temporal_covariate, axis=1)
        return df

    @property
    def temporal_coordinates(self):
        raise NotImplementedError

    def nb_steps(self, split: Split = Split.all) -> int:
        return len(self.df_temporal_coordinates(split))

    def df_temporal_range(self, split: Split = Split.all) -> Tuple[int, int]:
        df_temporal_coordinates = self.df_temporal_coordinates(split)
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

    def spatio_temporal_shape(self, split: Split.all) -> Tuple[int, int]:
        return len(self.df_spatial_coordinates(split)), len(self.df_temporal_coordinates(split))

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

    def __eq__(self, other):
        return self.df_merged.equals(other.df_merged)

    def __str__(self):
        return pd.concat([self.df_coordinates(), self.df_coordinate_climate_model], axis=1).__str__()
