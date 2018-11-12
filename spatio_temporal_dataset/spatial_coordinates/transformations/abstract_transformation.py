import pandas as pd


class AbstractTransformation(object):

    def __init__(self, nb_dimensions):
        self.nb_dimensions = nb_dimensions

    def transform(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        assert len(df_coord.columns) == self.nb_dimensions, "columns={}, nb_dimensions={}".format(df_coord.columns,
                                                                                                  self.nb_dimensions)
        return df_coord
