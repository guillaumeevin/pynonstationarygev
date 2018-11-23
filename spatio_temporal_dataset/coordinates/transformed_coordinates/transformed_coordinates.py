from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.abstract_transformation import AbstractTransformation


class TransformedCoordinates(AbstractSpatialCoordinates):

    @classmethod
    def from_coordinates(cls, spatial_coordinates: AbstractSpatialCoordinates,
                         transformation_function: AbstractTransformation):
        df_coordinates_transformed = spatial_coordinates.df_coordinates.copy()
        df_coordinates_transformed = transformation_function.transform(df_coord=df_coordinates_transformed)
        return cls(df_coordinates=df_coordinates_transformed, s_split=spatial_coordinates.s_split)


