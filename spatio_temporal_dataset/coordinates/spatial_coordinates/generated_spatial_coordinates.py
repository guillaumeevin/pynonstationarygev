import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rpy2.robjects import r

from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates


class CircleSpatialCoordinates(AbstractSpatialCoordinates):

    @classmethod
    def df_spatial(cls, nb_points, max_radius=1.0, random=True):
        # Sample uniformly inside the circle
        angles = np.array(r.runif(nb_points, max=2 * math.pi)) if random else np.linspace(0.0, 2 * math.pi, nb_points+1)[:-1]
        radius = np.sqrt(np.array(r.runif(nb_points, max=max_radius))) if random else np.ones(nb_points) * max_radius
        df = pd.DataFrame.from_dict({cls.COORDINATE_X: radius * np.cos(angles),
                                     cls.COORDINATE_Y: radius * np.sin(angles)})
        return df

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, max_radius=1.0, random=True):
        return cls.from_df(cls.df_spatial(nb_points, max_radius, random), train_split_ratio)

    def visualization_2D(self):
        radius = 1.0
        circle1 = plt.Circle((0, 0), radius, color='r', fill=False)
        plt.gcf().gca().set_xlim((-radius, radius))
        plt.gcf().gca().set_ylim((-radius, radius))
        plt.gcf().gca().add_artist(circle1)
        super().visualization_2D()


class CircleSpatialCoordinatesRadius2(CircleSpatialCoordinates):

    @classmethod
    def from_nb_points(cls, nb_points, train_split_ratio: float = None, max_radius=1.0, random=True):
        return 2 * super().from_nb_points(nb_points, train_split_ratio, max_radius, random)
