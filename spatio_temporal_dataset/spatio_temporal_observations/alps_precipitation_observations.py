import os.path as op

from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from utils import get_full_path


class AlpsPrecipitationObservations(AbstractSpatioTemporalObservations):
    RELATIVE_PATH = r'local/spatio_temporal_datasets/Gilles  - precipitations'
    FULL_PATH = get_full_path(relative_path=RELATIVE_PATH)

    @classmethod
    def from_csv(cls, csv_file='max_precip_3j'):
        csv_path = op.join(cls.FULL_PATH, csv_file + '.csv')
        return super().from_csv(csv_path)

    @classmethod
    def transform_gilles_txt_into_csv(cls):
        filepath = op.join(cls.FULL_PATH, 'original data', 'max_precip_3j.txt')
        station_to_coordinates = {}
        # todo: be absolutely sure of the mapping, in the txt file of Gilles we do not have the coordinate
        # this seems rather dangerous, especially if he loaded
        # todo: re-do the whole analysis, and extraction myself about the snow
        with open(filepath, 'r') as f:
            for l in f:
                _, station_name, coordinates = l.split('"')
                coordinates = coordinates.split()
                print(len(coordinates))
                assert len(coordinates) == 55
                station_to_coordinates[station_name] = coordinates
        # df = pd.DataFrame.from_dict(data=station_to_coordinates, orient='index',
        #                             columns=[cls.COORDINATE_X, cls.COORDINATE_Y, cls.COORDINATE_Z])
        # print(df.head())
        # filepath = op.join(cls.FULL_PATH, 'max_precip_3j.csv')
        # assert not op.exists(filepath)
        # df.to_csv(filepath)
