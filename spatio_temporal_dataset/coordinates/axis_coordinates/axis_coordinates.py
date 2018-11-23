import pandas as pd

import numpy as np

from extreme_estimator.extreme_models.utils import get_loaded_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractSpatialCoordinates


class AxisCoordinates(AbstractSpatialCoordinates):
    pass


class UniformAxisCoordinates(AxisCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, start=0.0, end=1.0):
        # Sample uniformly inside the circle
        r = get_loaded_r()
        axis_coordinates = np.array(r.runif(nb_points, min=start, max=end))
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: axis_coordinates})
        return cls.from_df(df)
