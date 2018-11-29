import pandas as pd
import numpy as np

from spatio_temporal_dataset.spatio_temporal_split import SpatialTemporalSplit, SpatioTemporalSlicer, \
    train_ind_from_s_split, TEST_SPLIT_STR, TRAIN_SPLIT_STR, s_split_from_ratio, spatio_temporal_slice


class AbstractSpatioTemporalObservations(object):

    def __init__(self, df_maxima_frech: pd.DataFrame = None, df_maxima_gev: pd.DataFrame = None,
                 s_split: pd.Series = None, train_split_ratio: float = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index
        Columns are the temporal moment of the maxima
        """
        assert df_maxima_frech is not None or df_maxima_gev is not None
        self.df_maxima_frech = df_maxima_frech
        self.df_maxima_gev = df_maxima_gev

        if s_split is not None and train_split_ratio is not None:
            raise AttributeError('A split is already defined, there is no need to specify a ratio')
        elif s_split is not None or train_split_ratio is not None:
            if train_split_ratio:
                s_split = s_split_from_ratio(length=self.nb_obs, train_split_ratio=train_split_ratio)
            assert s_split.isin([TRAIN_SPLIT_STR, TEST_SPLIT_STR]).all()
        self.s_split = s_split

    @property
    def nb_obs(self):
        if self.df_maxima_frech is not None:
            return len(self.df_maxima_frech.columns)
        else:
            return len(self.df_maxima_gev.columns)

    @classmethod
    def from_df(cls, df):
        pass

    def maxima_gev(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all, slicer: SpatioTemporalSlicer = None):
        return spatio_temporal_slice(self.df_maxima_gev, split, slicer).values

    def maxima_frech(self, split: SpatialTemporalSplit = SpatialTemporalSplit.all, slicer: SpatioTemporalSlicer = None):
        return spatio_temporal_slice(self.df_maxima_frech, split, slicer).values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: SpatialTemporalSplit = SpatialTemporalSplit.all,
                         slicer: SpatioTemporalSlicer = None):
        df = spatio_temporal_slice(self.df_maxima_frech, split, slicer)
        df.loc[:] = maxima_frech_values

    @property
    def train_ind(self) -> pd.Series:
        return train_ind_from_s_split(s_split=self.s_split)
