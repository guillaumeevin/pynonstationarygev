import pandas as pd
import numpy as np

from spatio_temporal_dataset.slicer.abstract_slicer import slice, AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split, \
    train_ind_from_s_split, TEST_SPLIT_STR, TRAIN_SPLIT_STR, s_split_from_ratio


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
                s_split = s_split_from_ratio(index=self._df_maxima.columns, train_split_ratio=train_split_ratio)
            assert len(s_split) == len(self._df_maxima.columns)
            assert s_split.isin([TRAIN_SPLIT_STR, TEST_SPLIT_STR]).all()
        self.s_split = s_split

    @property
    def _df_maxima(self) -> pd.DataFrame:
        if self.df_maxima_frech is not None:
            return self.df_maxima_frech
        else:
            return self.df_maxima_gev

    @property
    def index(self) -> pd.Index:
        return self._df_maxima.index

    @property
    def nb_obs(self) -> int:
        return len(self._df_maxima.columns)

    @classmethod
    def from_df(cls, df):
        pass

    def maxima_gev(self, split: Split = Split.all, slicer: AbstractSlicer = None) -> np.ndarray:
        return slice(self.df_maxima_gev, split, slicer).values

    def maxima_frech(self, split: Split = Split.all, slicer: AbstractSlicer = None) -> np.ndarray:
        return slice(self.df_maxima_frech, split, slicer).values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all,
                         slicer: AbstractSlicer = None):
        df = slice(self.df_maxima_frech, split, slicer)
        df.loc[:] = maxima_frech_values

    @property
    def train_ind(self) -> pd.Series:
        return train_ind_from_s_split(s_split=self.s_split)
