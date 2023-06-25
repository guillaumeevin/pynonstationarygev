import pandas as pd
import numpy as np
from sklearn.preprocessing import Normalizer, StandardScaler


class AbstractTransformation(object):

    def __init__(self, df_coordinates):
        self.df_coordinates = df_coordinates.copy()

    # todo: we could add some checks on the type of the input data
    @property
    def nb_dimensions(self):
        return self.df_coordinates.shape[1]

    def transform_float(self, coordinate: float):
        return self.transform_array(np.array([coordinate]))[0]

    def transform_array(self, coordinate: np.ndarray):
        assert len(coordinate) == self.nb_dimensions, "coordinate={}, nb_dimensions={}".format(coordinate,
                                                                                               self.nb_dimensions)

    def transform_serie(self, s_coord: pd.Series) -> pd.Series:
        return pd.Series(self.transform_array(s_coord.values), index=s_coord.index)

    def transform_df(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        return df_coord.copy().apply(self.transform_serie, axis=1)


class IdentityTransformation(AbstractTransformation):

    def transform_array(self, coordinate: np.ndarray):
        super().transform_array(coordinate)
        return coordinate

