from typing import List

import unittest

from extreme_estimator.extreme_models.margin_model.linear_margin_model import ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.slicer.split import ALL_SPLITS_EXCEPT_ALL, Split
from test.test_utils import load_test_1D_and_2D_spatial_coordinates, load_test_spatiotemporal_coordinates, \
    load_test_temporal_coordinates


class TestSlicerForDataset(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = None

    nb_points = 2
    nb_steps = 2
    nb_obs = 2

    @property
    def complete_shape(self):
        pass

    def load_datasets(self, train_split_ratio) -> List[AbstractDataset]:
        pass

    def get_shape(self, dataset, split):
        return dataset.maxima_frech(split).shape

    def check_shapes(self, train_split_ratio_to_observation_shape):
        assert self.complete_shape is not None
        for train_split_ratio, data_shape in train_split_ratio_to_observation_shape.items():
            for dataset in self.load_datasets(train_split_ratio):
                self.assertEqual(self.complete_shape, self.get_shape(dataset, Split.all))
                for split in ALL_SPLITS_EXCEPT_ALL:
                    if split in dataset.slicer.splits:
                        self.assertEqual(data_shape, self.get_shape(dataset, split))
                    else:
                        with self.assertRaises(AssertionError):
                            self.get_shape(dataset, split)


class TestSlicerForSpatialDataset(TestSlicerForDataset):

    @property
    def complete_shape(self):
        return self.nb_points, self.nb_obs

    def load_datasets(self, train_split_ratio):
        coordinates_list = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points,
                                                                   train_split_ratio=train_split_ratio)
        dataset_list = [FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs,
                                                                  margin_model=ConstantMarginModel(
                                                                      coordinates=coordinates),
                                                                  coordinates=coordinates, max_stable_model=Smith())
                        for coordinates in coordinates_list]
        return dataset_list

    def test_spatial_slicer_for_spatial_dataset(self):
        train_split_ratio_to_observation_shape = {
            None: self.complete_shape,
            0.5: (self.nb_points // 2, self.nb_obs),
        }
        self.check_shapes(train_split_ratio_to_observation_shape)


class TestSlicerForTemporalDataset(TestSlicerForDataset):

    @property
    def complete_shape(self):
        return self.nb_steps, self.nb_obs

    def load_datasets(self, train_split_ratio):
        coordinates_list = load_test_temporal_coordinates(nb_steps=self.nb_steps,
                                                          train_split_ratio=train_split_ratio)
        dataset_list = [FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs,
                                                                  margin_model=ConstantMarginModel(
                                                                      coordinates=coordinates),
                                                                  coordinates=coordinates, max_stable_model=Smith())
                        for coordinates in coordinates_list]
        return dataset_list

    def test_temporal_slicer_for_temporal_dataset(self):
        ind_tuple_to_observation_shape = {
            None: self.complete_shape,
            0.5: (self.nb_steps // 2, self.nb_obs),
        }
        self.check_shapes(ind_tuple_to_observation_shape)


class TestSlicerForSpatioTemporalDataset(TestSlicerForDataset):

    @property
    def complete_shape(self):
        return self.nb_points * self.nb_steps, self.nb_obs

    def load_datasets(self, train_split_ratio):
        coordinates_list = load_test_spatiotemporal_coordinates(nb_points=self.nb_points,
                                                                nb_steps=self.nb_steps,
                                                                train_split_ratio=train_split_ratio)
        coordinates_list = [coordinates for coordinates in coordinates_list if coordinates.nb_coordinates <= 2]
        dataset_list = [FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs,
                                                                  margin_model=ConstantMarginModel(
                                                                      coordinates=coordinates),
                                                                  coordinates=coordinates, max_stable_model=Smith())
                        for coordinates in coordinates_list]
        return dataset_list

    def test_spatiotemporal_slicer_for_spatio_temporal_dataset(self):
        ind_tuple_to_observation_shape = {
            None: self.complete_shape,
            0.5: (self.nb_steps * self.nb_points // 4, self.nb_obs),
        }
        self.check_shapes(ind_tuple_to_observation_shape)


if __name__ == '__main__':
    unittest.main()
