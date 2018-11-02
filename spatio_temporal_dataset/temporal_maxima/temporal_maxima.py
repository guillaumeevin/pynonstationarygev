import pandas as pd

class TemporalMaxima(object):

    def __init__(self, df_maxima: pd.DataFrame):
        """
        Main attribute of the class is the DataFrame df_maxima
        Index are stations index, Columns are the year of the maxima
        """
        self.df_maxima = df_maxima

    @classmethod
    def from_df(cls, df):
        pass

    @property
    def index(self):
        return self.df_maxima.index

    @property
    def maxima(self):
        return self.df_maxima.values

    # todo: add the transformed_maxima and the not-trasnformed maxima
