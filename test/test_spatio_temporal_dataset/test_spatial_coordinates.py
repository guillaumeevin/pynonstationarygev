import unittest

from spatio_temporal_dataset.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestSpatialCoordinates(unittest.TestCase):

    DISPLAY = False

    def test_circle(self):
        coord_2D = CircleCoordinatesRadius1.from_nb_points(nb_points=500)
        if self.DISPLAY:
            coord_2D.visualization_2D()
        self.assertTrue(True)

    def test_anisotropy(self):
        coord_3D = AlpsStation3DCoordinatesWithAnisotropy.from_csv()
        if self.DISPLAY:
            coord_3D.visualization_3D()
        self.assertTrue(True)

    def test_normalization(self):
        coord_2D = AlpsStation2DCoordinatesBetweenZeroAndOne.from_csv()
        if self.DISPLAY:
            coord_2D.visualization_2D()
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
