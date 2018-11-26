import os.path as op
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class AbstractCoordinates(object):
    # Columns
    COORDINATE_X = 'coord_x'
    COORDINATE_Y = 'coord_y'
    COORDINATE_Z = 'coord_z'
    COORDINATE_NAMES = [COORDINATE_X, COORDINATE_Y, COORDINATE_Z]
    COORD_SPLIT = 'coord_split'
    # Constants
    TRAIN_SPLIT_STR = 'train_split'
    TEST_SPLIT_STR = 'test_split'

    def __init__(self, df_coordinates: pd.DataFrame, s_split: pd.Series = None):
        self.df_coordinates = df_coordinates
        self.s_split = s_split

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        #  X and coordinates must be defined
        assert cls.COORDINATE_X in df.columns
        df_coordinates = df.loc[:, cls.coordinates_columns(df)]
        # Potentially, a split column can be specified
        s_split = df[cls.COORD_SPLIT] if cls.COORD_SPLIT in df.columns else None
        return cls(df_coordinates=df_coordinates, s_split=s_split)

    @classmethod
    def coordinates_columns(cls, df_coord: pd.DataFrame) -> List[str]:
        coord_columns = [cls.COORDINATE_X]
        for additional_coord in [cls.COORDINATE_Y, cls.COORDINATE_Z]:
            if additional_coord in df_coord.columns:
                coord_columns.append(additional_coord)
        return coord_columns

    @property
    def columns(self):
        return self.coordinates_columns(df_coord=self.df_coordinates)

    @property
    def nb_columns(self):
        return len(self.columns)

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
        coordinates = cls.from_csv()  # type: AbstractCoordinates
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

    def _coordinates_values(self, df_coordinates: pd.DataFrame) -> np.ndarray:
        return df_coordinates.loc[:, self.coordinates_columns(df_coordinates)].values

    @property
    def coordinates_values(self) -> np.ndarray:
        return self._coordinates_values(df_coordinates=self.df_coordinates)

    @property
    def x_coordinates(self) -> np.ndarray:
        return self.df_coordinates.loc[:, self.COORDINATE_X].values.copy()

    @property
    def y_coordinates(self) -> np.ndarray:
        return self.df_coordinates.loc[:, self.COORDINATE_Y].values.copy()

    @property
    def coordinates_train(self) -> np.ndarray:
        return self._coordinates_values(df_coordinates=self.df_coordinates_split(self.TRAIN_SPLIT_STR))

    @property
    def coordinates_test(self) -> np.ndarray:
        return self._coordinates_values(df_coordinates=self.df_coordinates_split(self.TEST_SPLIT_STR))

    @property
    def index(self):
        return self.df_coordinates.index

    #  Visualization

    def visualize(self):
        nb_coordinates_columns = len(self.coordinates_columns(self.df_coordinates))
        if nb_coordinates_columns == 1:
            self.visualization_1D()
        elif nb_coordinates_columns == 2:
            self.visualization_2D()
        else:
            self.visualization_3D()

    def visualization_1D(self):
        assert len(self.coordinates_columns(self.df_coordinates)) >= 1
        x = self.coordinates_values[:]
        y = np.zeros(len(x))
        plt.scatter(x, y)
        plt.show()

    def visualization_2D(self):
        assert len(self.coordinates_columns(self.df_coordinates)) >= 2
        x, y = self.coordinates_values[:, 0], self.coordinates_values[:, 1]
        plt.scatter(x, y)
        plt.show()

    def visualization_3D(self):
        assert len(self.coordinates_columns(self.df_coordinates)) == 3
        x, y, z = self.coordinates_values[:, 0], self.coordinates_values[:, 1], self.coordinates_values[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')  # type: Axes3D
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
