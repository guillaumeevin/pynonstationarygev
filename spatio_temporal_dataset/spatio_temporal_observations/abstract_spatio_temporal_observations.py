import os.path as op
import pandas as pd
import numpy as np

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.abstract_slicer import df_sliced, AbstractSlicer
from spatio_temporal_dataset.slicer.split import Split


class AbstractSpatioTemporalObservations(object):
    # Observation columns
    OBSERVATIONS_GEV = 'obs_gev'
    OBSERVATIONS_FRECH = 'obs_frech'

    def __init__(self, df_maxima_frech: pd.DataFrame = None, df_maxima_gev: pd.DataFrame = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are coordinates index
        Columns are independent observations from the same coordinates index
        """
        assert df_maxima_gev is not None or df_maxima_frech is not None
        if df_maxima_gev is not None and df_maxima_frech is not None:
            assert pd.Index.equals(df_maxima_gev.index, df_maxima_frech.index)
        self.df_maxima_gev = df_maxima_gev  # type: pd.DataFrame
        self.df_maxima_frech = df_maxima_frech  # type: pd.DataFrame

    @classmethod
    def from_csv(cls, csv_path: str = None):
        assert csv_path is not None
        assert op.exists(csv_path)
        df = pd.read_csv(csv_path, index_col=0)
        return cls.from_df(df)

    @property
    def _df_maxima(self) -> pd.DataFrame:
        if self.df_maxima_frech is not None:
            return self.df_maxima_frech
        else:
            return self.df_maxima_gev

    @property
    def df_maxima_merged(self) -> pd.DataFrame:
        df_maxima_list = []
        for df, suffix in [(self.df_maxima_gev, self.OBSERVATIONS_GEV),
                           (self.df_maxima_frech, self.OBSERVATIONS_FRECH)]:
            if df is not None:
                df_maxima = df.copy()
                df_maxima.columns = [str(c) + ' ' + suffix for c in df_maxima.columns]
                df_maxima_list.append(df_maxima)
        return pd.concat(df_maxima_list, axis=1)

    @property
    def index(self) -> pd.Index:
        return self._df_maxima.index

    @property
    def nb_obs(self) -> int:
        return len(self._df_maxima.columns)

    @classmethod
    def from_df(cls, df):
        df_maxima_list = []
        for suffix in [cls.OBSERVATIONS_GEV, cls.OBSERVATIONS_FRECH]:
            columns_with_suffix = [c for c in df.columns if str(c).endswith(suffix)]
            if columns_with_suffix:
                df_maxima = df[columns_with_suffix] if columns_with_suffix else None
                df_maxima.columns = [c.replace(' ' + suffix, '') for c in df_maxima.columns]
            else:
                df_maxima = None
            df_maxima_list.append(df_maxima)
        df_maxima_gev, df_maxima_frech = df_maxima_list
        if df_maxima_gev is not None and df_maxima_frech is not None:
            assert pd.Index.equals(df_maxima_gev.columns, df_maxima_frech.columns)
        return cls(df_maxima_gev=df_maxima_gev, df_maxima_frech=df_maxima_frech)

    def convert_to_spatio_temporal_index(self, coordinates: AbstractCoordinates):
        assert coordinates.has_spatio_temporal_coordinates
        assert len(coordinates.index) == len(self.index) * self.nb_obs
        assert pd.Index.equals(self.index, coordinates.spatial_index())
        self.df_maxima_frech = self.flatten_df(self.df_maxima_frech, coordinates.index)
        self.df_maxima_gev = self.flatten_df(self.df_maxima_gev, coordinates.index)

    @staticmethod
    def flatten_df(df, index):
        if df is not None:
            return pd.DataFrame(np.concatenate([df[c].values for c in df.columns]), index=index)

    def maxima_gev(self, split: Split = Split.all, slicer: AbstractSlicer = None) -> np.ndarray:
        return df_sliced(self.df_maxima_gev, split, slicer).values

    def maxima_frech(self, split: Split = Split.all, slicer: AbstractSlicer = None) -> np.ndarray:
        return df_sliced(self.df_maxima_frech, split, slicer).values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray, split: Split = Split.all,
                         slicer: AbstractSlicer = None):
        df = df_sliced(self.df_maxima_frech, split, slicer)
        df.loc[:] = maxima_frech_values
