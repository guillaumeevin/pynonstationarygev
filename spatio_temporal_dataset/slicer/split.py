from enum import Enum

import pandas as pd


class Split(Enum):
    all = 0
    # SpatioTemporal splits
    train_spatiotemporal = 1
    test_spatiotemporal = 2
    test_spatiotemporal_spatial = 3
    test_spatiotemporal_temporal = 4
    # Spatial splits
    train_spatial = 5
    test_spatial = 6
    # Temporal splits
    train_temporal = 7
    test_temporal = 8


ALL_SPLITS_EXCEPT_ALL = [split for split in Split if split is not Split.all]

SPLIT_NAME = 'split'
TRAIN_SPLIT_STR = 'train_split'
TEST_SPLIT_STR = 'test_split'


def train_ind_from_s_split(s_split):
    if s_split is None:
        return None
    else:
        return s_split.isin([TRAIN_SPLIT_STR])


def s_split_from_ratio(index, train_split_ratio):
    length = len(index)
    assert 0 < train_split_ratio < 1
    s = pd.Series(TEST_SPLIT_STR, index=index)
    nb_points_train = int(length * train_split_ratio)
    assert 0 < nb_points_train < length
    train_ind = pd.Series.sample(s, n=nb_points_train).index
    assert 0 < len(train_ind) < length, "number of training points:{} length:{}".format(len(train_ind), length)
    s.loc[train_ind] = TRAIN_SPLIT_STR
    return s
