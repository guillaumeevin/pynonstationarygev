import os.path as op
import pandas as pd
import numpy as np

from spatio_temporal_dataset.slicer.abstract_slicer import df_sliced, AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split

class AbstractSpatioTemporalObservations(object):

    def __init__(self, df_maxima_frech: pd.DataFrame = None, df_maxima_gev: pd.DataFrame = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index
        Columns are the temporal moment of the maxima
        """
        assert df_maxima_frech is not None or df_maxima_gev is not None
        self.df_maxima_frech = df_maxima_frech
        self.df_maxima_gev = df_maxima_gev

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path)
        # # Index correspond to the first column
        # index_column_name = df.columns[0]
        # assert index_column_name not in cls.coordinates_columns(df)
        # df.set_index(index_column_name, inplace=True)
        return cls.from_df(df)

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
        return df_sliced(self.df_maxima_gev, split, slicer).values

    def maxima_frech(self, split: Split = Split.all, slicer: AbstractSlicer = None) -> np.ndarray:
        return df_sliced(self.df_maxima_frech, split, slicer).values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all,
                         slicer: AbstractSlicer = None):
        df = df_sliced(self.df_maxima_frech, split, slicer)
        df.loc[:] = maxima_frech_values
