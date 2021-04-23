from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation \
    import AbstractTransformation


class TransformedCoordinates(AbstractCoordinates):

    @classmethod
    def from_coordinates(cls, coordinates: AbstractCoordinates,
                         transformation_class):
        df_coordinates = coordinates.df_all_coordinates.copy()
        transformation = transformation_class(df_coordinates)  # type: AbstractTransformation
        df_coordinates_transformed = transformation.transform_df(df_coordinates)
        return cls(df=df_coordinates_transformed)


