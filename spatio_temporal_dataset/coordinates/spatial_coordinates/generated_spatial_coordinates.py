import math
import numpy as np
import pandas as pd
from rpy2.robjects import r

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
import matplotlib.pyplot as plt


class CircleCoordinates(AbstractCoordinates):

    @classmethod
    def df_spatial(cls, nb_points, max_radius=1.0):
        # Sample uniformly inside the circle
        angles = np.array(r.runif(nb_points, max=2 * math.pi))
        radius = np.sqrt(np.array(r.runif(nb_points, max=max_radius)))
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: radius * np.cos(angles),
                                     cls.COORDINATE_Y: radius * np.sin(angles)})
        return df

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, max_radius=1.0):
        return cls.from_df(cls.df_spatial(nb_points, max_radius), train_split_ratio)

    def visualization_2D(self):
        r = 1.0
        circle1 = plt.Circle((0, 0), r, color='r', fill=False)
        plt.gcf().gca().set_xlim((-r, r))
        plt.gcf().gca().set_ylim((-r, r))
        plt.gcf().gca().add_artist(circle1)
        super().visualization_2D()


class CircleCoordinatesRadius2(CircleCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, max_radius=1.0):
        return 2 * super().from_nb_points(nb_points, train_split_ratio, max_radius)
