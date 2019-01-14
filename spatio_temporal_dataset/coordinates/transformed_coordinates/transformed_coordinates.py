from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation \
    import AbstractTransformation


class TransformedCoordinates(AbstractCoordinates):

    @classmethod
    def from_coordinates(cls, coordinates: AbstractCoordinates,
                         transformation_function: AbstractTransformation):
        df_coordinates_transformed = coordinates.df_all_coordinates.copy()
        df_coordinates_transformed = transformation_function.transform(df_coord=df_coordinates_transformed)
        return cls(df=df_coordinates_transformed, slicer_class=type(coordinates.slicer),
                   s_split_spatial=coordinates.s_split_spatial, s_split_temporal=coordinates.s_split_temporal)


