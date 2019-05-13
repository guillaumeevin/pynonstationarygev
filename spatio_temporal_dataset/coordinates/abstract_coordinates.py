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
    # Coordinates columns
    COORDINATES_NAMES = COORDINATE_SPATIAL_NAMES + [COORDINATE_T]
    # Coordinate type
    COORDINATE_TYPE = 'float64'

    def __init__(self, df: pd.DataFrame, slicer_class: type, s_split_spatial: pd.Series = None,
                 s_split_temporal: pd.Series = None, transformation_class: type = None):
        # Extract df_all_coordinates from df
        coordinate_columns = [c for c in df.columns if c in self.COORDINATES_NAMES]
        assert len(coordinate_columns) > 0
        # Sort coordinates according to a specified order
        sorted_coordinates_columns = [c for c in self.COORDINATES_NAMES if c in coordinate_columns]
        self.df_all_coordinates = df.loc[:, sorted_coordinates_columns].copy()  # type: pd.DataFrame
        # Cast df_all_coordinates to the desired type
        self.df_all_coordinates = self.df_all_coordinates.astype(self.COORDINATE_TYPE)

        # Slicing attributes
        self.s_split_spatial = s_split_spatial  # type: pd.Series
        self.s_split_temporal = s_split_temporal  # type: pd.Series
        self.slicer = None  # type: Union[None, AbstractSlicer]

        # Transformation attribute
        if transformation_class is None:
            transformation_class = IdentityTransformation
        # Transformation class is instantiated with all coordinates
        self.transformation = transformation_class(self.df_all_coordinates)
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

        # Infer the slicer class
        if s_split_temporal is None and s_split_spatial is None:
            raise ValueError('Both split are unspecified')
        elif s_split_temporal is not None and s_split_spatial is None:
            slicer_class = TemporalSlicer
        elif s_split_temporal is None and s_split_spatial is not None:
            slicer_class = SpatialSlicer
        else:
            slicer_class = SpatioTemporalSlicer

        return cls(df=df, slicer_class=slicer_class, s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal)

    @classmethod
    def from_df_and_slicer(cls, df: pd.DataFrame, slicer_class: type, train_split_ratio: float = None,
                           transformation_class: type = None):
        # All the index should be unique
        assert len(set(df.index)) == len(df), 'df indices are not unique'

        # Create a spatial split
        s_split_spatial = s_split_from_df(df, cls.COORDINATE_X, cls.SPATIAL_SPLIT, train_split_ratio, True)
        # Create a temporal split
        s_split_temporal = s_split_from_df(df, cls.COORDINATE_T, cls.TEMPORAL_SPLIT, train_split_ratio, False)

        return cls(df=df, slicer_class=slicer_class, s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal,
                   transformation_class=transformation_class)

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
        return self.df_all_coordinates.join(self.df_split)

    # Normalize

    def transform(self, coordinate: np.ndarray) -> np.ndarray:
        return self.transformation.transform_array(coordinate=coordinate)

    # Split

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        df_transformed_coordinates = self.transformation.transform_df(df_coord=self.df_all_coordinates)
        return df_sliced(df=df_transformed_coordinates, split=split, slicer=self.slicer)

    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.df_coordinates(split).values

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
        return self.coordinates_spatial_names + self.coordinates_temporal_names

    @property
    def nb_coordinates(self) -> int:
        return len(self.coordinates_names)

    @property
    def coordinates_dims(self) -> List[int]:
        return list(range(self.nb_coordinates))

    # Spatial attributes

    @property
    def coordinates_spatial_names(self) -> List[str]:
        return [name for name in self.COORDINATE_SPATIAL_NAMES if name in self.df_all_coordinates.columns]

    @property
    def nb_coordinates_spatial(self) -> int:
        return len(self.coordinates_spatial_names)

    @property
    def has_spatial_coordinates(self) -> bool:
        return self.nb_coordinates_spatial > 0

    def df_spatial_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        if self.nb_coordinates_spatial == 0:
            return pd.DataFrame()
        else:
            return self.df_coordinates(split).loc[:, self.coordinates_spatial_names].drop_duplicates()

    @property
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
    def coordinates_temporal_names(self) -> List[str]:
        return [self.COORDINATE_T] if self.COORDINATE_T in self.df_all_coordinates else []

    @property
    def nb_coordinates_temporal(self) -> int:
        return len(self.coordinates_temporal_names)

    @property
    def has_temporal_coordinates(self) -> bool:
        return self.nb_coordinates_temporal > 0

    def df_temporal_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        if self.nb_coordinates_temporal == 0:
            return pd.DataFrame()
        else:
            return self.df_coordinates(split).loc[:, self.coordinates_temporal_names].drop_duplicates()

    @property
    def nb_steps(self, split: Split = Split.all) -> int:
        return len(self.df_temporal_coordinates(split))

    def df_temporal_range(self, split: Split = Split.all) -> Tuple[int, int]:
        df_temporal_coordinates = self.df_temporal_coordinates(split)
        return int(df_temporal_coordinates.min()), int(df_temporal_coordinates.max()),

    @property
    def idx_temporal_coordinates(self):
        return self.coordinates_names.index(self.COORDINATE_T)

    # Spatio temporal attributes

    @property
    def has_spatio_temporal_coordinates(self) -> bool:
        return self.has_spatial_coordinates and self.has_temporal_coordinates

    def spatio_temporal_shape(self, split: Split.all) -> Tuple[int, int]:
        return len(self.df_spatial_coordinates(split)), len(self.df_temporal_coordinates(split))

    def ind_of_df_all_coordinates(self, coordinate_name, value):
        return self.df_all_coordinates.loc[:, coordinate_name] == value

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
        if self.nb_coordinates_spatial == 1:
            self.visualization_1D()
        elif self.nb_coordinates_spatial == 2:
            self.visualization_2D()
        else:
            self.visualization_3D()

    def visualization_1D(self):
        assert self.nb_coordinates_spatial >= 1
        x = self.x_coordinates
        y = np.zeros(len(x))
        plt.scatter(x, y)
        plt.show()

    def visualization_2D(self):
        assert self.nb_coordinates_spatial >= 2
        plt.scatter(self.x_coordinates, self.y_coordinates)
        plt.show()

    def visualization_3D(self):
        assert self.nb_coordinates_spatial == 3
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
        return self.df_coordinates().__str__()
