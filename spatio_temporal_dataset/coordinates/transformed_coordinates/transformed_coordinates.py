from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.abstract_transformation import AbstractTransformation


class TransformedCoordinates(AbstractCoordinates):

    @classmethod
    def from_coordinates(cls, coordinates: AbstractCoordinates,
                         transformation_function: AbstractTransformation):
        df_coordinates_transformed = coordinates.df_coordinates.copy()
        df_coordinates_transformed = transformation_function.transform(df_coord=df_coordinates_transformed)
        return cls(df_coordinates=df_coordinates_transformed, s_split=coordinates.s_split)


