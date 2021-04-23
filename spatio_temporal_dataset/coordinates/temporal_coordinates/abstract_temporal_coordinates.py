import pandas as pd
import numpy as np
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalCoordinates(AbstractCoordinates):

    @property
    def temporal_coordinates(self):
        return self

    @property
    def transformed_distance_between_two_successive_years(self):
        return self.transformation.transform_array(np.ones([1])) - self.transformation.transform_array(np.zeros([1]))

    @classmethod
    def from_df(cls, df: pd.DataFrame, transformation_class: type = None):
        assert cls.COORDINATE_T in df.columns
        assert not any([coordinate_name in df.columns for coordinate_name in cls.COORDINATE_SPATIAL_NAMES])
        return super().from_df_and_transformation_class(df, transformation_class)
