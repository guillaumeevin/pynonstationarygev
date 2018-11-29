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
            msg = "One split was not defined \n \n" \
                  "index: \n {}  \n, column:\n {} \n".format(self.index_train_ind, self.column_train_ind)
            assert self.index_train_ind is None and self.column_train_ind is None, msg

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
        assert pd.RangeIndex.equals(df.columns, self.column_train_ind.index)
        assert pd.RangeIndex.equals(df.index, self.index_train_ind.index)
        if split is SpatialTemporalSplit.train:
            return df.loc[self.index_train_ind, self.column_train_ind]
        elif split is SpatialTemporalSplit.test:
            return df.loc[self.index_test_ind, self.column_test_ind]
        elif split is SpatialTemporalSplit.test_spatial:
            return df.loc[self.index_test_ind, self.column_train_ind]
        elif split is SpatialTemporalSplit.test_temporal:
            return df.loc[self.index_train_ind, self.column_test_ind]


SPLIT_NAME = 'split'
TRAIN_SPLIT_STR = 'train_split'
TEST_SPLIT_STR = 'test_split'


def train_ind_from_s_split(s_split):
    """

    :param s_split:
    :return:
    """
    if s_split is None:
        return None
    else:
        return s_split.isin([TRAIN_SPLIT_STR])


def s_split_from_ratio(length, train_split_ratio):
    assert 0 < train_split_ratio < 1
    s = pd.Series([TEST_SPLIT_STR for _ in range(length)])
    nb_points_train = int(length * train_split_ratio)
    train_ind = pd.Series.sample(s, n=nb_points_train).index
    s.loc[train_ind] = TRAIN_SPLIT_STR
    return s
