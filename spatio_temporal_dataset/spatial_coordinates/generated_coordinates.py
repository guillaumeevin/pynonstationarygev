import math
import numpy as np
import pandas as pd

from extreme_estimator.R_fit.utils import get_loaded_r
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates
import matplotlib.pyplot as plt


class CircleCoordinatesRadius1(AbstractSpatialCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, max_radius=1.0):
        # Sample uniformly inside the circle
        r = get_loaded_r()
        angles = np.array(r.runif(nb_points, max=2 * math.pi))
        radius = np.sqrt(np.array(r.runif(nb_points, max=max_radius)))
        df = pd.DataFrame.from_dict({cls.COORD_X: radius * np.cos(angles), cls.COORD_Y: radius * np.sin(angles)})
        return cls.from_df(df)

    def visualization_2D(self):
        r = 1.0
        circle1 = plt.Circle((0, 0), r, color='r', fill=False)
        plt.gcf().gca().set_xlim((-r, r))
        plt.gcf().gca().set_ylim((-r, r))
        plt.gcf().gca().add_artist(circle1)
        super().visualization_2D()


class CircleCoordinatesRadius2(CircleCoordinatesRadius1):

    @classmethod
    def from_nb_points(cls, nb_points, max_radius=1.0):
        return 2 * super().from_nb_points(nb_points, max_radius)

