import pandas as pd
import numpy as np


class AbstractTransformation(object):

    def __init__(self, df_coordinates):
        self.df_coordinates = df_coordinates

    @property
    def nb_dimensions(self):
        return self.df_coordinates.shape[1]

    def transform_array(self, coordinate: np.ndarray):
        assert len(coordinate) == self.nb_dimensions, "coordinate={}, nb_dimensions={}".format(coordinate,
                                                                                               self.nb_dimensions)

    def transform_serie(self, s_coord: pd.Series) -> pd.Series:
        return pd.Series(self.transform_array(s_coord.values), index=s_coord.index)

    def transform_df(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        data = [self.transform_serie(r) for _, r in df_coord.iterrows()]
        return pd.DataFrame(data, index=df_coord.index, columns=df_coord.columns)


class IdentityTransformation(AbstractTransformation):

    def transform_array(self, coordinate: np.ndarray):
        super().transform_array(coordinate)
        return coordinate
