from enum import Enum

import pandas as pd


class SpatialTemporalSplit(Enum):
    all = 0
    train = 1
    test = 2
    test_temporal = 3
    test_spatial = 4


class SpatioTemporalSlicer(object):

    def __init__(self, coordinate_train_ind: pd.Series, observation_train_ind: pd.Series):
        self.index_train_ind = coordinate_train_ind  # type: pd.Series
        self.column_train_ind = observation_train_ind  # type: pd.Series
        if self.ind_are_not_defined:
            assert self.index_train_ind is None and self.column_train_ind is None, "One split was not defined"

    @property
    def index_test_ind(self) -> pd.Series:
        return ~self.index_train_ind

    @property
    def column_test_ind(self) -> pd.Series:
        return ~self.column_train_ind

    @property
    def ind_are_not_defined(self):
        return self.index_train_ind is None or self.column_train_ind is None

    def loc_split(self, df: pd.DataFrame, split: SpatialTemporalSplit):
        assert isinstance(split, SpatialTemporalSplit)
        # By default, if one of the two split is not defined we return all the data
        if self.ind_are_not_defined or split is SpatialTemporalSplit.all:
            return df
        assert df.columns == self.column_train_ind.index
        assert df.index == self.index_train_ind.index
        if split is SpatialTemporalSplit.train:
            return df.loc[self.index_train_ind, self.column_train_ind]
        elif split is SpatialTemporalSplit.test:
            return df.loc[self.index_test_ind, self.column_test_ind]
        elif split is SpatialTemporalSplit.test_spatial:
            return df.loc[self.index_test_ind, self.column_train_ind]
        elif split is SpatialTemporalSplit.test_temporal:
            return df.loc[self.index_train_ind, self.column_test_ind]
