import unittest

from spatio_temporal_dataset.coordinates.axis_coordinates.axis_coordinates import UniformAxisCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.coordinates.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestSpatialCoordinates(unittest.TestCase):
    DISPLAY = False

    def test_circle(self):
        coord = CircleCoordinatesRadius1.from_nb_points(nb_points=500)
        if self.DISPLAY:
            coord.visualization_2D()
        self.assertTrue(True)

    def test_anisotropy(self):
        coord = AlpsStation3DCoordinatesWithAnisotropy.from_csv()
        if self.DISPLAY:
            coord.visualization_3D()
        self.assertTrue(True)

    def test_normalization(self):
        coord = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()
        if self.DISPLAY:
            coord.visualization_2D()
        self.assertTrue(True)


class TestAxisCoordinates(unittest.TestCase):
    DISPLAY = False

    def test_unif(self):
        coord = UniformAxisCoordinates.from_nb_points(nb_points=10)
        if self.DISPLAY:
            coord.visualization_1D()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
