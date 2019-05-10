# Suffix to differentiate between spatio temporal index and spatial index
import pandas as pd
import numpy as np


def get_index_suffix(df_spatial: pd.DataFrame, t):
    index_type = type(df_spatial.index[0])
    assert index_type in [int, float, str, np.int64, np.float64], index_type
    return index_type(t * len(df_spatial))


def get_index_with_spatio_temporal_index_suffix(df_spatial: pd.DataFrame, t):
    index_suffix = get_index_suffix(df_spatial, t)
    return pd.Index([i + index_suffix for i in df_spatial.index])


def get_index_without_spatio_temporal_index_suffix(df_spatial: pd.DataFrame):
    index_suffix = get_index_suffix(df_spatial, 0)
    if isinstance(index_suffix, str):
        return df_spatial.index.str.split(index_suffix).str.join('')
    else:
        return df_spatial.index - index_suffix

