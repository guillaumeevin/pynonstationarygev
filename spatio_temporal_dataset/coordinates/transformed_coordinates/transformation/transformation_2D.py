from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation \
    import AbstractTransformation
import pandas as pd


class Transformation2D(AbstractTransformation):

    def __init__(self):
        super().__init__(nb_dimensions=2)


class Uniform2DNormalization(Transformation2D):
    """Normalize similarly the X and Y axis with a single function so as to conserve proportional distances"""

    def transform(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        df_coord = super().transform(df_coord)
        for i in range(2):
            df_coord.iloc[:, i] = self.uniform_normalization(df_coord.iloc[:, i])
        return df_coord

    def uniform_normalization(self, s_coord: pd.Series) -> pd.Series:
        return s_coord


class BetweenZeroAndOne2DNormalization(Uniform2DNormalization):
    """Normalize such that min(coord) >= (0,0) and max(coord) <= (1,1)"""

    def __init__(self) -> None:
        super().__init__()
        self.min_coord = None
        self.max_coord = None

    def transform(self, df_coord: pd.DataFrame) -> pd.DataFrame:
        # Compute the min and max globally
        self.min_coord, self.max_coord = df_coord.min().min(), df_coord.max().max()
        #  Then, call the super method that will call the uniform_normalization method
        return super().transform(df_coord)

    def uniform_normalization(self, s_coord: pd.Series) -> pd.Series:
        s_coord_shifted = s_coord - self.min_coord
        s_coord_scaled = s_coord_shifted / (self.max_coord - self.min_coord)
        return s_coord_scaled


class BetweenMinusOneAndOne2DNormalization(BetweenZeroAndOne2DNormalization):

    def uniform_normalization(self, s_coord: pd.Series) -> pd.Series:
        pass
