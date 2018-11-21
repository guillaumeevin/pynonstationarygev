from spatio_temporal_dataset.spatial_coordinates.alps_station_3D_coordinates import AlpsStation3DCoordinates
from spatio_temporal_dataset.spatial_coordinates.transformations.transformation_2D import \
    BetweenZeroAndOne2DNormalization
from spatio_temporal_dataset.spatial_coordinates.transformed_coordinates import TransformedCoordinates


class AlpsStation2DCoordinates(AlpsStation3DCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        # Remove the Z coordinates from df_coord
        spatial_coordinates = super().from_csv(csv_file)  # type: AlpsStation3DCoordinates
        spatial_coordinates.df_coordinates.drop(cls.COORDINATE_Z, axis=1, inplace=True)
        return spatial_coordinates


class AlpsStation2DCoordinatesBetweenZeroAndOne(AlpsStation2DCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        coord = super().from_csv(csv_file)
        return TransformedCoordinates.from_coordinates(spatial_coordinates=coord,
                                                       transformation_function=BetweenZeroAndOne2DNormalization())


class AlpsStationCoordinatesBetweenZeroAndTwo(AlpsStation2DCoordinatesBetweenZeroAndOne):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        return 2 * super().from_csv(csv_file)
