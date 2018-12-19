import os.path as op
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer, df_sliced
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from spatio_temporal_dataset.slicer.split import s_split_from_df, ind_train_from_s_split, Split
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer


class AbstractCoordinates(object):
    """

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

    def __init__(self, df_coord: pd.DataFrame, slicer_class: type, s_split_spatial: pd.Series = None,
                 s_split_temporal: pd.Series = None):
        self.df_all_coordinates = df_coord  # type: pd.DataFrame
        self.s_split_spatial = s_split_spatial  # type: pd.Series
        self.s_split_temporal = s_split_temporal  # type: pd.Series
        self.slicer = None  # type: AbstractSlicer

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
        # Extract df_coordinate
        coordinate_columns = [c for c in df.columns if c in cls.COORDINATES_NAMES]
        df_coord = df.loc[:, coordinate_columns].copy()

        # Extract the split
        split_columns = [c for c in df.columns if c in [cls.SPATIAL_SPLIT, cls.TEMPORAL_SPLIT]]
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

        # Remove all the columns used from df
        columns_used = coordinate_columns + split_columns
        df.drop(columns_used, axis=1, inplace=True)
        return cls(df_coord=df_coord, slicer_class=slicer_class,
                   s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal)

    @classmethod
    def from_df_and_slicer(cls, df: pd.DataFrame, slicer_class: type, train_split_ratio: float = None):
        # All the index should be unique
        assert len(set(df.index)) == len(df)

        # Create a spatial split
        s_split_spatial = s_split_from_df(df, cls.COORDINATE_X, cls.SPATIAL_SPLIT, train_split_ratio, True)
        # Create a temporal split
        s_split_temporal = s_split_from_df(df, cls.COORDINATE_T, cls.TEMPORAL_SPLIT, train_split_ratio, False)

        return cls(df_coord=df, slicer_class=slicer_class,
                   s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal)

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

    # Split

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        return df_sliced(df=self.df_all_coordinates, split=split, slicer=self.slicer)

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
    def coordinates_spatial_names(self) -> List[str]:
        return [name for name in self.COORDINATE_SPATIAL_NAMES if name in self.df_all_coordinates.columns]

    @property
    def nb_coordinates_spatial(self) -> int:
        return len(self.coordinates_spatial_names)

    @property
    def coordinates_temporal_names(self) -> List[str]:
        return [self.COORDINATE_T] if self.COORDINATE_T in self.df_all_coordinates else []

    @property
    def nb_coordinates_temporal(self) -> int:
        return len(self.coordinates_temporal_names)

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