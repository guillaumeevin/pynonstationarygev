from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformed_coordinates import TransformedCoordinates


class AlpsStation2DCoordinates(AlpsStation3DCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        # Remove the Z coordinates from df_coord
        spatial_coordinates = super().from_csv(csv_file)  # type: AlpsStation3DCoordinates
        spatial_coordinates.df_all_coordinates.drop(cls.COORDINATE_Z, axis=1, inplace=True)
        return spatial_coordinates


class AlpsStation2DCoordinatesBetweenZeroAndOne(AlpsStation2DCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        coord = super().from_csv(csv_file)
        return TransformedCoordinates.from_coordinates(coordinates=coord,
                                                       transformation_class=BetweenZeroAndOneNormalization)

class AlpsStationCoordinatesBetweenZeroAndTwo(AlpsStation2DCoordinatesBetweenZeroAndOne):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        return 2 * super().from_csv(csv_file)
