from enum import Enum
from typing import Union

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


def split_to_display_kwargs(split: Split):
    marker = None
    gridsize = 1000
    if 'train' in split.name:
        linewidth = 0.5
    else:
        linewidth = 2
        if 'spatiotemporal' in split.name:
            gridsize = 20
            if 'spatial' in split.name and 'temporal' in split.name:
                marker = '*'
            elif 'spatial' in split.name:
                marker = '^'
            else:
                marker = '>'
    return {'marker': marker, 'linewidth': linewidth, 'gridsize': gridsize}


ALL_SPLITS_EXCEPT_ALL = [split for split in Split if split is not Split.all]

SPLIT_NAME = 'split'
TRAIN_SPLIT_STR = 'train_split'
TEST_SPLIT_STR = 'test_split'


def invert_s_split(s_split):
    ind = ind_train_from_s_split(s_split)
    s_split.loc[ind] = TEST_SPLIT_STR
    s_split.loc[~ind] = TRAIN_SPLIT_STR
    return s_split


def ind_train_from_s_split(s_split):
    if s_split is None:
        return None
    else:
        return s_split.isin([TRAIN_SPLIT_STR])


def small_s_split_from_ratio(index: pd.Index, train_split_ratio):
    length = len(index)
    assert 0 < train_split_ratio < 1
    s = pd.Series(TEST_SPLIT_STR, index=index)
    nb_points_train = int(length * train_split_ratio)
    assert 0 < nb_points_train < length
    train_ind = pd.Series.sample(s, n=nb_points_train).index
    assert 0 < len(train_ind) < length, "number of training points:{} length:{}".format(len(train_ind), length)
    s.loc[train_ind] = TRAIN_SPLIT_STR
    return s


def s_split_from_df(df: pd.DataFrame, column, split_column, train_split_ratio, spatial_split) -> Union[None, pd.Series]:
    df = df.copy()  # type: pd.DataFrame
    # Extract the index
    if train_split_ratio is None:
        return None
    if column not in df:
        return None
    elif split_column in df:
        raise Exception('A split has already been defined')
    else:
        s = df.drop_duplicates(subset=[column], keep='first')[column]
        assert len(df) % len(s) == 0
        multiplication_factor = len(df) // len(s)
        small_s_split = small_s_split_from_ratio(s.index, train_split_ratio)
        if spatial_split:
            # concatenation for spatial_split
            s_split = pd.concat([small_s_split for _ in range(multiplication_factor)], ignore_index=True).copy()
        else:
            # dilatation for the temporal split
            s_split = pd.Series(None, index=df.index)
            for i in range(len(s_split)):
                s_split.iloc[i] = small_s_split.iloc[i // multiplication_factor]
        s_split.index = df.index
        return s_split
