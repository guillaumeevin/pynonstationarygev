import unittest

from spatio_temporal_dataset.spatial_coordinates.alps_station_2D_coordinates import \
    AlpsStation2DCoordinatesBetweenZeroAndOne
from spatio_temporal_dataset.spatial_coordinates.alps_station_3D_coordinates import \
    AlpsStation3DCoordinatesWithAnisotropy
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestTemporalObservations(unittest.TestCase):

    DISPLAY = False

if __name__ == '__main__':
    unittest.main()
