import pandas as pd

import numpy as np
from rpy2.robjects import r

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractUniDimensionalCoordinates(AbstractCoordinates):
    pass


class LinSpaceCoordinates(AbstractUniDimensionalCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, start=-1.0, end=1.0):
        axis_coordinates = np.linspace(start, end, nb_points)
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
        return cls.from_df(df, train_split_ratio)


class UniformCoordinates(AbstractUniDimensionalCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, start=-1.0, end=1.0):
        # Sample uniformly inside the circle
        axis_coordinates = np.array(r.runif(nb_points, min=start, max=end))
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
        return cls.from_df(df, train_split_ratio)
