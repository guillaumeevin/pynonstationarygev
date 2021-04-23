import numpy as np
import pandas as pd
from rpy2.robjects import r

from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates


class AbstractUniDimensionalSpatialCoordinates(AbstractSpatialCoordinates):
    pass


class LinSpaceSpatialCoordinates(AbstractUniDimensionalSpatialCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points: float = None, start=-1.0, end=1.0, **kwargs):
        axis_coordinates = np.linspace(start, end, nb_points)
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
        return cls.from_df(df, **kwargs)


class UniformSpatialCoordinates(AbstractUniDimensionalSpatialCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, start=-1.0, end=1.0, **kwargs):
        # Sample uniformly inside the circle
        df = cls.df_spatial(nb_points, start, end)
        return cls.from_df(df, **kwargs)

    @classmethod
    def df_spatial(cls, nb_points, start=-1.0, end=1.0):
        axis_coordinates = np.array(r.runif(nb_points, min=start, max=end))
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
        return df
