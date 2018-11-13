import pandas as pd


class AbstractTemporalObservations(object):

    def __init__(self, df_maxima_frech: pd.DataFrame = None, df_maxima_gev: pd.DataFrame = None):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index
        Columns are the temporal moment of the maxima
        """
        self.df_maxima_frech = df_maxima_frech
        self.df_maxima_gev = df_maxima_gev

    @classmethod
    def from_df(cls, df):
        pass

    @property
    def maxima_gev(self):
        return self.df_maxima_gev.values

    @property
    def maxima_frech(self):
        return self.df_maxima_frech.values

    @maxima_frech.setter
    def maxima_frech(self, maxima_frech_to_set):
        assert maxima_frech_to_set is not None
        assert maxima_frech_to_set.shape == self.maxima_gev.shape
        self.df_maxima_frech = pd.DataFrame(data=maxima_frech_to_set,
                                            index=self.df_maxima_gev.index,
                                            columns=self.df_maxima_gev.columns)

    @property
    def column_to_time_index(self):
        pass

    @property
    def index(self):
        return self.df_maxima_gev.index



