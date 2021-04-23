import numpy as np
import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates


class AbstractSpatioTemporalObservations(object):
    # Observation columns
    OBSERVATIONS_GEV = 'obs_gev'
    OBSERVATIONS_FRECH = 'obs_frech'

    def __init__(self, df_maxima_gev: pd.DataFrame = None, df_maxima_frech: pd.DataFrame = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are coordinates index
        Columns are independent observations from the same coordinates index

        For example, if we have spatial coordinates,
        then all the columns might correspond to annual maxima observations for different years

        If we have a spatio-temporal coordinates,
        then all the columns might correspond to observations that were made at some spatial coodinate, and for some given year
        """
        assert df_maxima_gev is not None or df_maxima_frech is not None
        assert isinstance(df_maxima_gev, pd.DataFrame) or isinstance(df_maxima_frech, pd.DataFrame)
        if df_maxima_gev is not None and df_maxima_frech is not None:
            assert pd.Index.equals(df_maxima_gev.index, df_maxima_frech.index)
        self.df_maxima_gev = df_maxima_gev  # type: pd.DataFrame
        self.df_maxima_frech = df_maxima_frech  # type: pd.DataFrame

    @property
    def _df_maxima(self) -> pd.DataFrame:
        if self.df_maxima_frech is not None:
            return self.df_maxima_frech
        else:
            return self.df_maxima_gev

    def normalize(self):
        print('normalize')
        # It should stay superior to 0 and lower or equal to 1
        # Thus the easiest way to do that is to divide by the maximum
        # maxima = self._df_maxima.values.flatten().max()
        # self._df_maxima /= maxima
        pass

    @property
    def df_maxima_merged(self) -> pd.DataFrame:
        df_maxima_list = []
        for df, suffix in [(self.df_maxima_gev, self.OBSERVATIONS_GEV),
                           (self.df_maxima_frech, self.OBSERVATIONS_FRECH)]:
            if df is not None:
                df_maxima = df.copy()
                df_maxima.columns = [str(c) for c in df_maxima.columns]
                df_maxima_list.append(df_maxima)
        return pd.concat(df_maxima_list, axis=1)

    @property
    def index(self) -> pd.Index:
        return self._df_maxima.index

    @property
    def columns(self) -> pd.Index:
        return self._df_maxima.columns

    @property
    def nb_obs(self) -> int:
        return len(self.columns)

    def convert_to_spatio_temporal_index(self, coordinates: AbstractCoordinates):
        assert coordinates.has_spatio_temporal_coordinates
        assert len(coordinates.index) == len(self.index) * self.nb_obs
        assert pd.Index.equals(self.index, coordinates.spatial_index)
        self.df_maxima_frech = self.flatten_df(self.df_maxima_frech, coordinates.index)
        self.df_maxima_gev = self.flatten_df(self.df_maxima_gev, coordinates.index)

    @staticmethod
    def flatten_df(df, index):
        if df is not None:
            return pd.DataFrame(np.concatenate([df[c].values for c in df.columns]), index=index)

    def convert_to_temporal_index(self, temporal_coordinates: AbstractTemporalCoordinates, spatial_idx: int):
        assert len(self.index) > len(temporal_coordinates) and len(self.index) % len(temporal_coordinates) == 0
        spatial_len = len(self.index) // len(temporal_coordinates)
        assert 0 <= spatial_idx < spatial_len
        # Build ind to select the observations of interest
        ind = np.zeros(spatial_len, dtype=bool)
        ind[spatial_idx] = True
        ind = np.concatenate([ind for _ in range(len(temporal_coordinates))])
        self.df_maxima_frech = self.loc_df(self.df_maxima_frech, ind, temporal_coordinates.index)
        self.df_maxima_gev = self.loc_df(self.df_maxima_gev, ind, temporal_coordinates.index)

    @staticmethod
    def loc_df(df, ind, new_index):
        if df is not None:
            df = df.loc[ind]
            df.index = new_index
            return df

    @property
    def maxima_gev(self) -> np.ndarray:
        return self.df_maxima_gev.values

    @property
    def maxima_frech(self) -> np.ndarray:
        return self.df_maxima_frech.values

    def set_maxima_frech(self, maxima_frech_values: np.ndarray):
        self.df_maxima_frech.loc[:] = maxima_frech_values

    def __str__(self) -> str:
        return self._df_maxima.__str__()

    def print_summary(self):
        # Write a summary of observations
        df = self.df_maxima_gev
        print('Observations summary:', '        ', end='')
        print('Mean value:', df.mean().mean(), '        ', end='')
        print('Min value:', df.min().min(), '        ', end='')
        print('Max value:', df.max().max(), '        ', end='')
        percentage = round(100 * (df.size - np.count_nonzero(df.values)) / df.size, 1)
        print('Percentage of zero values {} out of {} observations'.format(percentage, df.size), '\n')

    @_df_maxima.setter
    def _df_maxima(self, value):
        self.__df_maxima = value
