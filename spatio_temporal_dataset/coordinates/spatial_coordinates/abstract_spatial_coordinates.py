from abc import ABC

import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractSpatialCoordinates(AbstractCoordinates, ABC):

    @classmethod
    def from_list_x_coordinates(cls, x_coordinates, transformation_class: type = None):
        df = pd.DataFrame({cls.COORDINATE_X: x_coordinates})
        return cls.from_df(df, transformation_class)

    @classmethod
    def from_df(cls, df: pd.DataFrame, transformation_class: type = None):
        assert cls.COORDINATE_X in df.columns
        assert cls.COORDINATE_T not in df.columns
        return super().from_df_and_transformation_class(df, transformation_class)

    @classmethod
    def from_nb_points(cls, nb_points: int, **kwargs):
        # Call the default class method from csv
        coordinates = cls.from_csv()  # type: AbstractCoordinates
        # Check that nb_points asked is not superior to the number of coordinates
        nb_coordinates = len(coordinates)
        if nb_points > nb_coordinates:
            raise Exception('Nb coordinates in csv: {} < Nb points desired: {}'.format(nb_coordinates, nb_points))
        # Sample randomly nb_points coordinates
        df_sample = pd.DataFrame.sample(coordinates.df_coordinates(), n=nb_points)
        return cls.from_df(df=df_sample, **kwargs)
