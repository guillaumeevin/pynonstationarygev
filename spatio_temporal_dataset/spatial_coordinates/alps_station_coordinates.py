import pandas as pd
import os.path as op

from spatio_temporal_dataset.spatial_coordinates.abstract_coordinates import AbstractSpatialCoordinates
from spatio_temporal_dataset.spatial_coordinates.normalized_coordinates import BetweenZeroAndOneNormalization, \
    NormalizedCoordinates
from utils import get_full_path


class AlpsStationCoordinates(AbstractSpatialCoordinates):
    RELATIVE_PATH = r'local/spatio_temporal_datasets/Gilles  - precipitations'
    FULL_PATH = get_full_path(relative_path=RELATIVE_PATH)

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        csv_path = op.join(cls.FULL_PATH, csv_file + '.csv')
        return super().from_csv(csv_path)

    @classmethod
    def transform_txt_into_csv(cls):
        filepath = op.join(cls.FULL_PATH, 'original data', 'coord-lambert2.txt')
        station_to_coordinates = {}
        with open(filepath, 'r') as f:
            for l in f:
                _, station_name, coordinates = l.split('"')
                coordinates = coordinates.split()
                assert len(coordinates) == 3
                station_to_coordinates[station_name] = coordinates
        df = pd.DataFrame.from_dict(data=station_to_coordinates, orient='index',
                                    columns=[cls.COORD_X, cls.COORD_Y, cls.COOR_ID])
        df.to_csv(op.join(cls.FULL_PATH, 'coord-lambert2.csv'))
        print(df.head())
        print(df.index)


class AlpsStationCoordinatesBetweenZeroAndOne(AlpsStationCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        coord = super().from_csv(csv_file)
        return NormalizedCoordinates.from_coordinates(spatial_coordinates=coord,
                                                      normalizing_function=BetweenZeroAndOneNormalization())


if __name__ == '__main__':
    # AlpsStationCoordinate.transform_txt_into_csv()
    # coord = AlpsStationCoordinates.from_csv()
    # coord = AlpsStationCoordinates.from_nb_points(nb_points=60)
    # coord = AlpsStationCoordinatesBetweenZeroAndOne.from_csv()
    coord = AlpsStationCoordinatesBetweenZeroAndOne.from_nb_points(nb_points=60)
    coord.visualization()
