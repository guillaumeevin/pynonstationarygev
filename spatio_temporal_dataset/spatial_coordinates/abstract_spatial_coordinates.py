import os.path as op
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AbstractSpatialCoordinates(object):
    # Columns
    COORD_X = 'coord_x'
    COORD_Y = 'coord_y'
    COORD_Z = 'coord_z'
    COORD_SPLIT = 'coord_split'
    # Constants
    TRAIN_SPLIT_STR = 'train_split'
    TEST_SPLIT_STR = 'test_split'

    def __init__(self, df_coordinates: pd.DataFrame, s_split: pd.Series = None):
        self.df_coordinates = df_coordinates
        self.s_split = s_split

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        #  X and Y coordinates must be defined
        assert cls.COORD_X in df.columns and cls.COORD_Y in df.columns
        df_coordinates = df.loc[:, cls.coordinates_columns(df)]
        # Potentially, a split column can be specified
        s_split = df[cls.COORD_SPLIT] if cls.COORD_SPLIT in df.columns else None
        return cls(df_coordinates=df_coordinates, s_split=s_split)

    @classmethod
    def coordinates_columns(cls, df_coord: pd.DataFrame) -> List[str]:
        # If a Z coordinate is in the DataFrame, then
        coord_columns = [cls.COORD_X, cls.COORD_Y]
        if cls.COORD_Z in df_coord.columns:
            coord_columns.append(cls.COORD_Z)
        return coord_columns

    @property
    def columns(self):
        return self.coordinates_columns(df_coord=self.df_coordinates)

    @property
    def df(self) -> pd.DataFrame:
        # Merged DataFrame of df_coord and s_split
        return self.df_coordinates if self.s_split is None else self.df_coordinates.join(self.s_split)

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        return cls.from_df(df)

    @classmethod
    def from_nb_points(cls, nb_points: int, **kwargs):
        # Call the default class method from csv
        coordinates = cls.from_csv()  # type: AbstractSpatialCoordinates
        # Sample randomly nb_points coordinates
        nb_coordinates = len(coordinates)
        if nb_points > nb_coordinates:
            raise Exception('Nb coordinates in csv: {} < Nb points desired: {}'.format(nb_coordinates, nb_points))
        else:
            df_sample = pd.DataFrame.sample(coordinates.df, n=nb_points)
            return cls.from_df(df=df_sample)

    def df_coordinates_split(self, split_str: str) -> pd.DataFrame:
        assert self.s_split is not None
        ind = self.s_split == split_str
        return self.df_coordinates.loc[ind]

    def coordinates_values(self, df_coordinates: pd.DataFrame) -> np.ndarray:
        return df_coordinates.loc[:, self.coordinates_columns(df_coordinates)].values

    @property
    def coordinates(self) -> np.ndarray:
        return self.coordinates_values(df_coordinates=self.df_coordinates)

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.df_coordinates.loc[:, self.COORD_X].values.copy()

    @property
    def y_coordinates(self) -> np.ndarray:
        return self.df_coordinates.loc[:, self.COORD_Y].values.copy()

    @property
    def coordinates_train(self) -> np.ndarray:
        return self.coordinates_values(df_coordinates=self.df_coordinates_split(self.TRAIN_SPLIT_STR))

    @property
    def coordinates_test(self) -> np.ndarray:
        return self.coordinates_values(df_coordinates=self.df_coordinates_split(self.TEST_SPLIT_STR))

    @property
    def index(self):
        return self.df_coordinates.index

    #  Visualization

    def visualization_2D(self):
        x, y = self.coordinates[:, 0], self.coordinates[:, 1]
        plt.scatter(x, y)
        plt.show()

    def visualization_3D(self):
        assert len(self.coordinates_columns(self.df_coordinates)) == 3
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
        x, y, z = self.coordinates[:, 0], self.coordinates[:, 1], self.coordinates[:, 2]
        ax.scatter(x, y, z, marker='^')
        plt.show()

    #  Magic Methods

    def __len__(self):
        return len(self.df_coordinates)

    def __mul__(self, other: float):
        self.df_coordinates *= other
        return self

    def __rmul__(self, other):
        return self * other
