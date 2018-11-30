import os.path as op
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from spatio_temporal_dataset.slicer.split import s_split_from_ratio, TEST_SPLIT_STR, \
    TRAIN_SPLIT_STR, train_ind_from_s_split, Split
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer


class AbstractCoordinates(object):
    # Columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    COORDINATE_SPLIT = 'coord_split'

    def __init__(self, df_coord: pd.DataFrame, s_split: pd.Series = None):
        self.df_coord = df_coord  # type: pd.DataFrame
        self.s_split = s_split  # type: pd.Series

    # ClassMethod constructor

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None):
        #  X and coordinates must be defined
        assert cls.COORDINATE_X in df.columns
        # Create a split based on the train_split_ratio
        if train_split_ratio is not None:
            assert cls.COORDINATE_SPLIT not in df.columns, "A split has already been defined"
            s_split = s_split_from_ratio(index=df.index, train_split_ratio=train_split_ratio)
            df[cls.COORDINATE_SPLIT] = s_split
        # Potentially, a split column can be specified directly in df
        if cls.COORDINATE_SPLIT not in df.columns:
            df_coord = df
            s_split = None
        else:
            df_coord = df.loc[:, cls.coordinates_columns(df)]
            s_split = df[cls.COORDINATE_SPLIT]
            assert s_split.isin([TRAIN_SPLIT_STR, TEST_SPLIT_STR]).all()
        return cls(df_coord=df_coord, s_split=s_split)

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        # Index correspond to the first column
        index_column_name = df.columns[0]
        assert index_column_name not in cls.coordinates_columns(df)
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

    @classmethod
    def coordinates_columns(cls, df_coord: pd.DataFrame) -> List[str]:
        coord_columns = [cls.COORDINATE_X]
        for additional_coord in [cls.COORDINATE_Y, cls.COORDINATE_Z]:
            if additional_coord in df_coord.columns:
                coord_columns.append(additional_coord)
        return coord_columns

    @property
    def columns(self):
        return self.coordinates_columns(df_coord=self.df_coord)

    @property
    def nb_columns(self):
        return len(self.columns)

    @property
    def index(self):
        return self.df_coord.index

    @property
    def df_merged(self) -> pd.DataFrame:
        # Merged DataFrame of df_coord and s_split
        return self.df_coord if self.s_split is None else self.df_coord.join(self.s_split)

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        if self.train_ind is None:
            return self.df_coord
        if split is Split.all:
            return self.df_coord
        if split in [Split.train_temporal, Split.test_temporal]:
            return self.df_coord
        elif split in [Split.train_spatial, Split.train_spatiotemporal, Split.test_spatiotemporal_temporal]:
            return self.df_coord.loc[self.train_ind]
        elif split in [Split.test_spatial, Split.test_spatiotemporal, Split.test_spatiotemporal_spatial]:
            return self.df_coord.loc[~self.train_ind]
        else:
            raise NotImplementedError('Unknown split: {}'.format(split))

    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.df_coordinates(split).values

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.df_coord[self.COORDINATE_X].values.copy()

    @property
    def y_coordinates(self) -> np.ndarray:
        return self.df_coord[self.COORDINATE_Y].values.copy()

    @property
    def train_ind(self) -> pd.Series:
        return train_ind_from_s_split(s_split=self.s_split)

    #  Visualization

    def visualize(self):
        nb_coordinates_columns = len(self.coordinates_columns(self.df_coord))
        if nb_coordinates_columns == 1:
            self.visualization_1D()
        elif nb_coordinates_columns == 2:
            self.visualization_2D()
        else:
            self.visualization_3D()

    def visualization_1D(self):
        assert len(self.coordinates_columns(self.df_coord)) >= 1
        x = self.coordinates_values()[:]
        y = np.zeros(len(x))
        plt.scatter(x, y)
        plt.show()

    def visualization_2D(self):
        assert len(self.coordinates_columns(self.df_coord)) >= 2
        coordinates_values = self.coordinates_values()
        x, y = coordinates_values[:, 0], coordinates_values[:, 1]
        plt.scatter(x, y)
        plt.show()

    def visualization_3D(self):
        assert len(self.coordinates_columns(self.df_coord)) == 3
        coordinates_values = self.coordinates_values()
        x, y, z = coordinates_values[:, 0], coordinates_values[:, 1], coordinates_values[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
        ax.scatter(x, y, z, marker='^')
        plt.show()

    #  Magic Methods

    def __len__(self):
        return len(self.df_coord)

    def __mul__(self, other: float):
        self.df_coord *= other
        return self

    def __rmul__(self, other):
        return self * other
