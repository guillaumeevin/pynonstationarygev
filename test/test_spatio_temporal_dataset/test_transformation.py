import unittest

import numpy as np

from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization


class TestTransformation(unittest.TestCase):

    def test_temporal_normalization(self):
        nb_steps = 3
        start = 1950
        transformation_class_to_expected = {BetweenZeroAndOneNormalization: [0.0, 0.5, 1.0],
                                            BetweenMinusOneAndOneNormalization: [-1.0, 0.0, 1.0]}
        for transformation_class, expected in transformation_class_to_expected.items():
            temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=nb_steps,
                                                                                         start=start,
                                                                                         transformation_class=transformation_class)
            normalized_coordinates = temporal_coordinates.df_coordinates().iloc[:, 0].values
            expected_coordinates = np.array(expected)
            equals = normalized_coordinates == expected_coordinates
            self.assertTrue(equals.all(),
                            msg="expected: {}, res:{}".format(expected_coordinates, normalized_coordinates))


if __name__ == '__main__':
    unittest.main()
