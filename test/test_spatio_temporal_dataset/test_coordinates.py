import unittest
from collections import Counter

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    UniformSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.coordinates_1D import UniformSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleSpatialCoordinates
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        self.assertTrue(True)

    def test_unif(self):
        self.coord = UniformSpatialCoordinates.from_nb_points(nb_points=10)

    def test_circle(self):
        self.coord = CircleSpatialCoordinates.from_nb_points(nb_points=500)

    def test_normalization(self):
        self.coord = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()

    def test_anisotropy(self):
        self.coord = AlpsStation3DCoordinatesWithAnisotropy.from_csv()


class SpatioTemporalCoordinates(unittest.TestCase):
    nb_points = 4
    nb_times_steps = 2

    def tearDown(self):
        c = Counter([len(self.coordinates.df_coordinates(split)) for split in SpatioTemporalSlicer.SPLITS])
        good_count = c == Counter([2, 2, 2, 2]) or c == Counter([0, 0, 4, 4])
        self.assertTrue(good_count)

    def test_temporal_circle(self):
        self.coordinates = UniformSpatioTemporalCoordinates.from_nb_points(nb_points=self.nb_points,
                                                                           nb_time_steps=self.nb_times_steps,
                                                                           train_split_ratio=0.5)
    # def test_temporal_alps(self):
    #     pass


if __name__ == '__main__':
    unittest.main()
