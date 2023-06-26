import unittest
from typing import List

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from test.test_utils import load_test_1D_and_2D_spatial_coordinates, \
    load_test_max_stable_models


class TestMaxStableModel(unittest.TestCase):

    def setUp(self) -> None:
        self.nb_obs = 3
        self.nb_points = 2

    def test_rmaxstab_with_various_coordinates(self):
        smith_process = load_test_max_stable_models()[0]
        coordinates = load_test_1D_and_2D_spatial_coordinates(
            nb_points=self.nb_points)  # type: List[AbstractCoordinates]
        for coord in coordinates:
            res = smith_process.rmaxstab(nb_obs=self.nb_obs, coordinates_values=coord.coordinates_values(),
                                         use_rmaxstab_with_2_coordinates=True)
            self.assertEqual((self.nb_points, self.nb_obs), res.shape)

if __name__ == '__main__':
    unittest.main()
