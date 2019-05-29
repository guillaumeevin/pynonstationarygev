import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer


class AbstractSpatialCoordinates(AbstractCoordinates):

    @classmethod
    def from_df(cls, df: pd.DataFrame, train_split_ratio: float = None, transformation_class: type = None):
        assert cls.COORDINATE_X in df.columns
        assert cls.COORDINATE_T not in df.columns
        return super().from_df_and_slicer(df, SpatialSlicer, train_split_ratio, transformation_class)

    @classmethod
    def from_nb_points(cls, nb_points: int, train_split_ratio: float = None, **kwargs):
        # Call the default class method from csv
        coordinates = cls.from_csv()  # type: AbstractCoordinates
        # Check that nb_points asked is not superior to the number of coordinates
        nb_coordinates = len(coordinates)
        if nb_points > nb_coordinates:
            raise Exception('Nb coordinates in csv: {} < Nb points desired: {}'.format(nb_coordinates, nb_points))
        # Sample randomly nb_points coordinates
        df_sample = pd.DataFrame.sample(coordinates.df_merged, n=nb_points)
        return cls.from_df(df=df_sample, train_split_ratio=train_split_ratio, **kwargs)
