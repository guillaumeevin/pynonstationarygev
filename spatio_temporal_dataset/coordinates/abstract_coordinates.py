import os.path as op
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from spatio_temporal_dataset.slicer.abstract_slicer import AbstractSlicer, df_sliced
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from spatio_temporal_dataset.slicer.split import s_split_from_df, TEST_SPLIT_STR, \
    TRAIN_SPLIT_STR, ind_train_from_s_split, Split
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer


class AbstractCoordinates(object):
    # Spatial columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_SPATIAL_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    SPATIAL_SPLIT = 'spatial_split'
    # Temporal columns
    COORDINATE_T = 'coord_t'
    TEMPORAL_SPLIT = 'coord_temporal_split'
    COORDINATES_NAMES = COORDINATE_SPATIAL_NAMES + [COORDINATE_T]

    def __init__(self, df_coord: pd.DataFrame, s_split_spatial: pd.Series = None, s_split_temporal: pd.Series = None,
                 slicer_class: type = SpatialSlicer):
        self.df_all_coordinates = df_coord  # type: pd.DataFrame
        self.s_split_spatial = s_split_spatial  # type: pd.Series
        self.s_split_temporal = s_split_temporal  # type: pd.Series
        self.slicer = slicer_class(ind_train_spatial=self.ind_train_spatial,
                                   ind_train_temporal=self.ind_train_temporal)  # type: AbstractSlicer
        assert isinstance(self.slicer, AbstractSlicer)

    # ClassMethod constructor

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None, slicer_class: type = SpatialSlicer):
        """
        train_split_ratio is shared between the spatial part of the data, and the temporal part
        """
        # All the index should be unique
        assert len(set(df.index)) == len(df)

        # Create a spatial split
        s_split_spatial = s_split_from_df(df, cls.COORDINATE_X, cls.SPATIAL_SPLIT, train_split_ratio, concat=False)

        # Create a temporal split
        if slicer_class is SpatioTemporalSlicer:
            s_split_temporal = s_split_from_df(df, cls.COORDINATE_T, cls.TEMPORAL_SPLIT, train_split_ratio, concat=True)
        else:
            s_split_temporal = None

        return cls(df_coord=df, s_split_spatial=s_split_spatial, s_split_temporal=s_split_temporal,
                   slicer_class=slicer_class)

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

    @classmethod
    def from_nb_points(cls, nb_points: int, train_split_ratio: float = None, **kwargs):
        # Call the default class method from csv
        coordinates = cls.from_csv()  # type: AbstractCoordinates
        # Check that nb_points asked is not superior to the number of coordinates
        nb_coordinates = len(coordinates)
        if nb_points > nb_coordinates:
            raise Exception('Nb coordinates in csv: {} < Nb points desired: {}'.format(nb_coordinates, nb_points))
        # Sample randomly nb_points coordinates
        df_sample = pd.DataFrame.sample(coordinates.df_merged, n=nb_points)
        return cls.from_df(df=df_sample, train_split_ratio=train_split_ratio)

    @property
    def index(self):
        return self.df_all_coordinates.index

    @property
    def df_merged(self) -> pd.DataFrame:
        # Merged DataFrame of df_coord and s_split
        return self.df_all_coordinates if self.s_split_spatial is None else self.df_all_coordinates.join(
            self.s_split_spatial)

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

    # Columns

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
    def t_coordinates(self):
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
