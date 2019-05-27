import numpy as np

from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    AbstractTransformation


class MultipleTransformation(AbstractTransformation):

    def __init__(self, transformation_1, transformation_2):
        self.transformation_1 = transformation_1  # type: AbstractTransformation
        self.transformation_2 = transformation_2  # type: AbstractTransformation

    @property
    def nb_dimensions(self):
        return self.transformation_1.nb_dimensions + self.transformation_2.nb_dimensions

    def transform_array(self, coordinate: np.ndarray):
        super().transform_array(coordinate)
        coordinate_1 = coordinate[:self.transformation_1.nb_dimensions]
        transformed_coordinate_1 = self.transformation_1.transform_array(coordinate_1)
        coordinate_2 = coordinate[-self.transformation_2.nb_dimensions:]
        transformed_coordinate_2 = self.transformation_2.transform_array(coordinate_2)
        transformed_coordinate = np.concatenate([transformed_coordinate_1, transformed_coordinate_2])
        return transformed_coordinate
