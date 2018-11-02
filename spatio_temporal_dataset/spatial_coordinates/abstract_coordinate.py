import os.path as op
import pandas as pd
import matplotlib.pyplot as plt


class AbstractSpatialCoordinates(object):

    # Columns
    COORD_X = 'coord_x'
    COORD_Y = 'coord_y'
    COORD_SPLIT = 'coord_split'
    # Constants
    TRAIN_SPLIT_STR = 'train_split'
    TEST_SPLIT_STR = 'test_split'

    def __init__(self, df_coord: pd.DataFrame, s_split: pd.Series = None):
        self.s_split = s_split
        self.df_coord = df_coord

    @classmethod
    def from_df(cls, df):
        assert cls.COORD_X in df.columns and cls.COORD_Y in df.columns
        df_coord = df.loc[:, [cls.COORD_X, cls.COORD_Y]]
        s_split = df[cls.COORD_SPLIT] if cls.COORD_SPLIT in df.columns else None
        return cls(df_coord=df_coord, s_split=s_split)

    @classmethod
    def from_csv(cls, csv_path):
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        return cls.from_df(df)

    def df_coord_split(self, split_str):
        assert self.s_split is not None
        ind = self.s_split == split_str
        return self.df_coord.loc[ind]

    @property
    def df_coord_train(self):
        return self.df_coord_split(self.TRAIN_SPLIT_STR)

    @property
    def df_coord_test(self):
        return self.df_coord_split(self.TEST_SPLIT_STR)

    @property
    def nb_points(self):
        return len(self.df_coord)

    @property
    def coord(self):
        return self.df_coord.values

    @property
    def index(self):
        return self.df_coord.index

    def visualization(self):
        x, y = self.coord[:, 0], self.coord[:, 1]
        plt.scatter(x, y)
        plt.show()
