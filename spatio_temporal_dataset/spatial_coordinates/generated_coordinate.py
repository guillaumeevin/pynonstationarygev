import math
import numpy as np
import pandas as pd

from extreme_estimator.R_fit.utils import get_loaded_r
from spatio_temporal_dataset.spatial_coordinates.abstract_coordinate import AbstractSpatialCoordinates
import matplotlib.pyplot as plt


class SimulatedCoordinates(AbstractSpatialCoordinates):
    """
    Common manipulation on generated coordinates
    """

    def __init__(self, df_coord, s_split=None):
        super().__init__(df_coord, s_split)

    @classmethod
    def from_nb_points(cls, nb_points, **kwargs):
        pass


class CircleCoordinates(SimulatedCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, **kwargs):
        max_radius = kwargs.get('max_radius', 1.0)
        # Sample uniformly inside the circle
        r = get_loaded_r()
        angles = np.array(r.runif(nb_points, max=2 * math.pi))
        radius = np.sqrt(np.array(r.runif(nb_points, max=max_radius)))
        df = pd.DataFrame.from_dict({cls.COORD_X: radius * np.cos(angles), cls.COORD_Y: radius * np.sin(angles)})
        return cls.from_df(df)

    def visualization(self):
        r = 1.0
        circle1 = plt.Circle((0, 0), r, color='r', fill=False)
        plt.gcf().gca().set_xlim((-r, r))
        plt.gcf().gca().set_ylim((-r, r))
        plt.gcf().gca().add_artist(circle1)
        super().visualization()


if __name__ == '__main__':
    coord = CircleCoordinates.from_nb_points(nb_points=500, max_radius=1)
    coord.visualization()
