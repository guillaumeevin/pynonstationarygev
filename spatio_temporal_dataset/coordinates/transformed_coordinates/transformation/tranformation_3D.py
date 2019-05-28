import pandas as pd
import numpy as np
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation \
    import AbstractTransformation
import math


class Transformation3D(AbstractTransformation):
    pass


class AnisotropyTransformation(Transformation3D):

    def __init__(self, df_coordinates, phi: float = 0.0, w1: float = 1.0, w2: float = 1.0):
        super().__init__(df_coordinates)
        """
        Anisotropy transformation
        :param phi: Between 0 and Pi, it corresponds to the angle of strongest dependence
        :param w1: > 0, it corresponds to the anisotropy ratio
        :param w2:  it corresponds to the weight for the altitude
        """
        self.phi = phi
        self.w1 = w1
        self.w2 = w2
        assert 0 <= self.phi < math.pi
        assert self.w1 > 0

    def transform_array(self, coordinate: np.ndarray):
        super().transform_array(coordinate)
        cosinus, sinus = math.cos(self.phi), math.sin(self.phi)
        inverse_w1 = 1 / self.w1
        V = np.array([
            [cosinus, -sinus, 0],
            [inverse_w1 * sinus, inverse_w1 * cosinus, 0],
            [0, 0, self.w2],
        ])
        coord_arr = np.transpose(coordinate)
        coord_arr = np.dot(V, coord_arr)
        return np.transpose(coord_arr)
