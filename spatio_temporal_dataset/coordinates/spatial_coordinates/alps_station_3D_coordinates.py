import os.path as op

import pandas as pd

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.tranformation_3D import AnisotropyTransformation
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformed_coordinates import TransformedCoordinates
from utils import get_full_path


class AlpsStation3DCoordinates(AbstractCoordinates):
    """
    For the Alps Stations, X, Y coordinates are in Lambert 2. Altitude is in meters
    """
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
                                    columns=[cls.COORDINATE_X, cls.COORDINATE_Y, cls.COORDINATE_Z])
        print(df.head())
        filepath = op.join(cls.FULL_PATH, 'coord-lambert2.csv')
        assert not op.exists(filepath)
        df.to_csv(filepath)


class AlpsStation3DCoordinatesWithAnisotropy(AlpsStation3DCoordinates):

    @classmethod
    def from_csv(cls, csv_file='coord-lambert2'):
        coord = super().from_csv(csv_file)
        return TransformedCoordinates.from_coordinates(coordinates=coord,
                                                       transformation_function=AnisotropyTransformation())
