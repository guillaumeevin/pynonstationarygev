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
    # Spatial columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    COORDINATE_SPATIAL_SPLIT = 'coord_spatial_split'
    # Temporal columns
    COORDINATE_T = 'coord_t'
    COORDINATE_TEMPORAL_SPLIT = 'coord_temporal_split'

    def __init__(self, df_coord: pd.DataFrame, s_spatial_split: pd.Series = None):
        self.df_all_coordinates = df_coord  # type: pd.DataFrame
        self.s_spatial_split = s_spatial_split  # type: pd.Series

    # ClassMethod constructor

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None):
        #  X and coordinates must be defined
        assert cls.COORDINATE_X in df.columns
        # Create a split based on the train_split_ratio
        if train_split_ratio is not None:
            assert cls.COORDINATE_SPATIAL_SPLIT not in df.columns, "A split has already been defined"
            s_split = s_split_from_ratio(index=df.index, train_split_ratio=train_split_ratio)
            df[cls.COORDINATE_SPATIAL_SPLIT] = s_split
        # Potentially, a split column can be specified directly in df
        if cls.COORDINATE_SPATIAL_SPLIT not in df.columns:
            df_coord = df
            s_split = None
        else:
            df_coord = df.loc[:, cls.coordinates_spatial_columns(df)]
            s_split = df[cls.COORDINATE_SPATIAL_SPLIT]
            assert s_split.isin([TRAIN_SPLIT_STR, TEST_SPLIT_STR]).all()
        return cls(df_coord=df_coord, s_spatial_split=s_split)

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        # Index correspond to the first column
        index_column_name = df.columns[0]
        assert index_column_name not in cls.coordinates_spatial_columns(df)
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
    def coordinates_spatial_columns(cls, df_coord: pd.DataFrame) -> List[str]:
        coord_columns = [cls.COORDINATE_X]
        for additional_coord in [cls.COORDINATE_Y, cls.COORDINATE_Z]:
            if additional_coord in df_coord.columns:
                coord_columns.append(additional_coord)
        return coord_columns

    @property
    def columns(self):
        return self.coordinates_spatial_columns(df_coord=self.df_all_coordinates)

    @property
    def nb_columns(self):
        return len(self.columns)

    @property
    def index(self):
        # todo: this should be replace whenever possible by coordinates_index
        return self.df_all_coordinates.index



    @property
    def df_merged(self) -> pd.DataFrame:
        # Merged DataFrame of df_coord and s_split
        return self.df_all_coordinates if self.s_spatial_split is None else self.df_all_coordinates.join(self.s_spatial_split)

    def df_coordinates(self, split: Split = Split.all) -> pd.DataFrame:
        if self.ind_train_spatial is None:
            return self.df_all_coordinates

        if split is Split.all:
            return self.df_all_coordinates

        if split in [Split.train_temporal, Split.test_temporal]:
            return self.df_all_coordinates

        elif split in [Split.train_spatial, Split.train_spatiotemporal, Split.test_spatiotemporal_temporal]:
            return self.df_all_coordinates.loc[self.ind_train_spatial]

        elif split in [Split.test_spatial, Split.test_spatiotemporal, Split.test_spatiotemporal_spatial]:
            return self.df_all_coordinates.loc[~self.ind_train_spatial]

        else:
            raise NotImplementedError('Unknown split: {}'.format(split))

    def coordinates_values(self, split: Split = Split.all) -> np.ndarray:
        return self.df_coordinates(split).values

    def coordinate_index(self, split: Split = Split.all) -> pd.Index:
        return self.df_coordinates(split).index

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_X].values.copy()

    @property
    def y_coordinates(self) -> np.ndarray:
        return self.df_all_coordinates[self.COORDINATE_Y].values.copy()

    @property
    def ind_train_spatial(self) -> pd.Series:
        return train_ind_from_s_split(s_split=self.s_spatial_split)

    #  Visualization

    def visualize(self):
        nb_coordinates_columns = len(self.coordinates_spatial_columns(self.df_all_coordinates))
        if nb_coordinates_columns == 1:
            self.visualization_1D()
        elif nb_coordinates_columns == 2:
            self.visualization_2D()
        else:
            self.visualization_3D()

    def visualization_1D(self):
        assert len(self.coordinates_spatial_columns(self.df_all_coordinates)) >= 1
        x = self.coordinates_values()[:]
        y = np.zeros(len(x))
        plt.scatter(x, y)
        plt.show()

    def visualization_2D(self):
        assert len(self.coordinates_spatial_columns(self.df_all_coordinates)) >= 2
        coordinates_values = self.coordinates_values()
        x, y = coordinates_values[:, 0], coordinates_values[:, 1]
        plt.scatter(x, y)
        plt.show()

    def visualization_3D(self):
        assert len(self.coordinates_spatial_columns(self.df_all_coordinates)) == 3
        coordinates_values = self.coordinates_values()
        x, y, z = coordinates_values[:, 0], coordinates_values[:, 1], coordinates_values[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
        ax.scatter(x, y, z, marker='^')
        plt.show()

    #  Magic Methods

    def __len__(self):
        return len(self.df_all_coordinates)

    def __mul__(self, other: float):
        self.df_all_coordinates *= other
        return self

    def __rmul__(self, other):
        return self * other
