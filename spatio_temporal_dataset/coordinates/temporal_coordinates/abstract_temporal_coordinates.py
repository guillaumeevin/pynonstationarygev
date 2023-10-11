import pandas as pd
import numpy as np
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalCoordinates(AbstractCoordinates):

    @property
    def temporal_coordinates(self):
        return self


    @classmethod
    def from_df(cls, df: pd.DataFrame):
        assert cls.COORDINATE_T in df.columns
        assert not any([coordinate_name in df.columns for coordinate_name in cls.COORDINATE_SPATIAL_NAMES])
        return super().from_df(df)
