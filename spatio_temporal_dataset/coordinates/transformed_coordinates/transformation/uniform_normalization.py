import numpy as np

from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation \
    import AbstractTransformation


class UniformNormalization(AbstractTransformation):
    """Normalize similarly the X and Y axis with a single function so as to conserve proportional distances"""

    def transform_array(self, coordinate: np.ndarray):
        super().transform_array(coordinate)
        for i in range(self.nb_dimensions):
            coordinate[i] = self.uniform_normalization(coordinate[i])
        return coordinate

    def uniform_normalization(self, coordinate_value: np.ndarray) -> np.ndarray:
        return coordinate_value


class BetweenZeroAndOneNormalization(UniformNormalization):
    """Normalize such that min(coord) >= (0,0) and max(coord) <= (1,1)"""

    def __init__(self, df_coordinates):
        super().__init__(df_coordinates)
        self.min_coord = self.df_coordinates.min().min()
        self.max_coord = self.df_coordinates.max().max()

    def uniform_normalization(self, coordinate_value: np.ndarray) -> np.ndarray:
        coord_shifted = coordinate_value - self.min_coord
        coord_scaled = coord_shifted / (self.max_coord - self.min_coord)
        return coord_scaled


class BetweenZeroAndTenNormalization(BetweenZeroAndOneNormalization):

    def uniform_normalization(self, coordinate_value: np.ndarray) -> np.ndarray:
        return super().uniform_normalization(coordinate_value) / 10


epsilon = 0.001


class BetweenZeroAndOneNormalizationMinEpsilon(BetweenZeroAndOneNormalization):

    def __init__(self, df_coordinates):
        super().__init__(df_coordinates)
        gap = self.max_coord - self.min_coord
        self.min_coord -= gap * epsilon


class BetweenZeroAndOneNormalizationMaxEpsilon(BetweenZeroAndOneNormalization):

    def __init__(self, df_coordinates):
        super().__init__(df_coordinates)
        gap = self.max_coord - self.min_coord
        self.max_coord += gap * epsilon


class BetweenMinusOneAndOneNormalization(BetweenZeroAndOneNormalization):
    """Normalize such that min(coord) >= (-1,-1) and max(coord) <= (1,1)"""

    def uniform_normalization(self, coordinate_value: np.ndarray) -> np.ndarray:
        coord = super().uniform_normalization(coordinate_value)
        coord *= 2
        coord -= 1
        return coord
