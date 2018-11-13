import pandas as pd


class AbstractTemporalObservations(object):

    def __init__(self, df_maxima_normalized: pd.DataFrame = None, df_maxima: pd.DataFrame = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index
        Columns are the temporal moment of the maxima
        """
        self.df_maxima_normalized = df_maxima_normalized
        self.df_maxima = df_maxima

    @classmethod
    def from_df(cls, df):
        pass

    @property
    def maxima(self):
        return self.df_maxima.values

    @property
    def maxima_normalized(self):
        return self.df_maxima_normalized.values

    @maxima_normalized.setter
    def maxima_normalized(self, maxima_normalized_to_set):
        assert self.df_maxima_normalized is None
        assert maxima_normalized_to_set is not None
        assert maxima_normalized_to_set.shape == self.maxima.shape
        self.df_maxima_normalized = pd.DataFrame(data=maxima_normalized_to_set, index=self.df_maxima.index,
                                                 columns=self.df_maxima.columns)

    @property
    def column_to_time_index(self):
        pass

    @property
    def index(self):
        return self.df_maxima.index


class RealTemporalObservations(object):

    def __init__(self):
        pass


class NormalizedTemporalObservations(object):
    pass
