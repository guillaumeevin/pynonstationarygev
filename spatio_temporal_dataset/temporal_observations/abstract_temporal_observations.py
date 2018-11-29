import pandas as pd
import numpy as np

from spatio_temporal_dataset.dataset.spatio_temporal_split import SpatialTemporalSplit, SpatioTemporalSlicer


class AbstractTemporalObservations(object):

    # Constants for the split column
    TRAIN_SPLIT_STR = 'train_split'
    TEST_SPLIT_STR = 'test_split'

    def __init__(self, df_maxima_frech: pd.DataFrame = None, df_maxima_gev: pd.DataFrame = None,
                 s_split: pd.Series = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index
        Columns are the temporal moment of the maxima
        """
        if s_split is not None:
            assert s_split.isin([self.TRAIN_SPLIT_STR, self.TEST_SPLIT_STR])
        self.s_split = s_split
        self.df_maxima_frech = df_maxima_frech
        self.df_maxima_gev = df_maxima_gev

    @classmethod
    def from_df(cls, df):
        pass

    @staticmethod
    def df_maxima(df: pd.DataFrame, split: SpatialTemporalSplit = SpatialTemporalSplit.all,
                  slicer: SpatioTemporalSlicer = None):
        if slicer is None:
            assert split is SpatialTemporalSplit.all
            return df
        else:
            return slicer.loc_split(df, split)

    def maxima_gev(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all, slicer: SpatioTemporalSlicer = None):
        return self.df_maxima(self.df_maxima_gev, split, slicer).values

    def maxima_frech(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all, slicer: SpatioTemporalSlicer = None):
        return self.df_maxima(self.df_maxima_frech, split, slicer).values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: SpatialTemporalSplit = SpatialTemporalSplit.all,
                         slicer: SpatioTemporalSlicer = None):
        df = self.df_maxima(self.df_maxima_frech, split, slicer)
        df.loc[:] = maxima_frech_values

    @property
    def train_ind(self) -> pd.Series:
        if self.s_split is None:
            return None
        else:
            return self.s_split.isin([self.TRAIN_SPLIT_STR])
