import unittest

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import UniformCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates


class TestCoordinates(unittest.TestCase):
    DISPLAY = False

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)
        self.coord = None  # type:  AbstractCoordinates

    def tearDown(self):
        if self.DISPLAY:
            self.coord.visualize()
        self.assertTrue(True)

    def test_unif(self):
        self.coord = UniformCoordinates.from_nb_points(nb_points=10)

    def test_circle(self):
        self.coord = CircleCoordinates.from_nb_points(nb_points=500)

    def test_normalization(self):
        self.coord = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()

    def test_anisotropy(self):
        self.coord = AlpsStation3DCoordinatesWithAnisotropy.from_csv()


if __name__ == '__main__':
    unittest.main()
