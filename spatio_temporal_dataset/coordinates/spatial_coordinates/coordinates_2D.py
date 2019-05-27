import numpy as np
import pandas as pd
from rpy2.robjects import r

from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import LinSpaceSpatialCoordinates


class AbstractBiDimensionalSpatialCoordinates(AbstractSpatialCoordinates):
    pass


class LinSpaceSpatial2DCoordinates(AbstractBiDimensionalSpatialCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, start=-1.0, end=1.0, **kwargs):
        df = cls.df_spatial(nb_points, start, end)
        return cls.from_df(df, train_split_ratio, **kwargs)

    @classmethod
    def df_spatial(cls, nb_points, start=-1.0, end=1.0):
        axis_coordinates = np.linspace(start, end, nb_points)
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates,
                                     cls.COORDINATE_Y: axis_coordinates})
        return df
