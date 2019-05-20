import unittest

import numpy as np

from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.generated_spatio_temporal_coordinates import \
    GeneratedSpatioTemporalCoordinates, UniformSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization, BetweenMinusOneAndOneNormalization


class TestTransformation(unittest.TestCase):

    def test_temporal_normalization(self):
        nb_steps = 3
        start = 1950
        transformation_class_to_expected = {
            BetweenZeroAndOneNormalization: [0.0, 0.5, 1.0],
            BetweenMinusOneAndOneNormalization: [-1.0, 0.0, 1.0],
            CenteredScaledNormalization: [-1.22474487, 0., 1.22474487],
        }
        for transformation_class, expected in transformation_class_to_expected.items():
            temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=nb_steps,
                                                                                         start=start,
                                                                                         transformation_class=transformation_class)
            normalized_coordinates = temporal_coordinates.df_coordinates().iloc[:, 0].values
            expected_coordinates = np.array(expected)
            equal = np.allclose(normalized_coordinates , expected_coordinates)
            self.assertTrue(equal, msg="expected: {}, res:{}".format(expected_coordinates, normalized_coordinates))

    def test_spatio_temporal_normalization(self):

        transformation_class_to_expected = {BetweenZeroAndOneNormalization: [0.0, 1.0],
                                            BetweenMinusOneAndOneNormalization: [-1.0, 1.0]}

        for transformation_class, expected in transformation_class_to_expected.items():
            coordinates = UniformSpatioTemporalCoordinates.from_nb_points_and_nb_steps(nb_points=2, nb_steps=50,
                                                                                       transformation_class=transformation_class)
            # Temporal coordinates, the order is known
            normalized_coordinates = coordinates.temporal_coordinates.df_coordinates().iloc[:, 0].values
            normalized_coordinates = np.array([normalized_coordinates[0], normalized_coordinates[-1]])
            expected_coordinates = np.array(expected)
            equals = normalized_coordinates == expected_coordinates
            self.assertTrue(equals.all(),
                            msg="expected: {}, res:{}".format(expected_coordinates, normalized_coordinates))
            # Spatial coordinates, we do not know the order
            normalized_coordinates = coordinates.temporal_coordinates.df_coordinates().iloc[:, 0].values
            normalized_coordinates = {normalized_coordinates[0], normalized_coordinates[-1]}
            expected_coordinates = set(expected)
            equals = normalized_coordinates == expected_coordinates
            self.assertTrue(equals,
                            msg="expected: {}, res:{}".format(expected_coordinates, normalized_coordinates))


if __name__ == '__main__':
    unittest.main()
