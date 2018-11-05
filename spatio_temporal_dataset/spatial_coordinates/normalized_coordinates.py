import pandas as pd

from spatio_temporal_dataset.spatial_coordinates.abstract_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.spatial_coordinates.alps_station_coordinates import AlpsStationCoordinate


class AbstractNormalizingFunction(object):

    def normalize(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        assert len(df_coord.columns) == 2
        return df_coord


class NormalizedCoordinates(AbstractSpatialCoordinates):

    @classmethod
    def from_coordinates(cls, spatial_coordinates: AbstractSpatialCoordinates,
                         normalizing_function: AbstractNormalizingFunction):
        df_coord_normalized = spatial_coordinates.df_coord.copy()
        coord_XY = [spatial_coordinates.COORD_X, spatial_coordinates.COORD_Y]
        df_coord_normalized.loc[:, coord_XY] = normalizing_function.normalize(df_coord_normalized.loc[:, coord_XY])
        return cls(df_coord=df_coord_normalized, s_split=spatial_coordinates.s_split)


"""
Define various types of normalizing functions
"""


class UniformNormalization(AbstractNormalizingFunction):
    """Normalize similarly the X and Y axis with a single function so as to conserve proportional distances"""

    def normalize(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        df_coord = super().normalize(df_coord)
        for i in range(2):
            df_coord.iloc[:, i] = self.uniform_normalization(df_coord.iloc[:, i])
        return df_coord

    def uniform_normalization(self, s_coord: pd.Series) -> pd.Series:
        return s_coord


class BetweenZeroAndOneNormalization(UniformNormalization):
    """Normalize such that min(coord) >= (0,0) and max(coord) <= (1,1)"""

    def __init__(self) -> None:
        self.min_coord = None
        self.max_coord = None

    def normalize(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        # Compute the min and max globally
        self.min_coord, self.max_coord = df_coord.min().min(), df_coord.max().max()
        #  Then, call the super method that will call the uniform_normalization method
        return super().normalize(df_coord)

    def uniform_normalization(self, s_coord: pd.Series) -> pd.Series:
        s_coord_shifted = s_coord - self.min_coord
        s_coord_scaled = s_coord_shifted / (self.max_coord - self.min_coord)
        return s_coord_scaled


if __name__ == '__main__':
    coord = AlpsStationCoordinate.from_csv()
    normalized_coord = NormalizedCoordinates.from_coordinates(spatial_coordinates=coord,
                                                              normalizing_function=BetweenZeroAndOneNormalization())
    normalized_coord.visualization()
