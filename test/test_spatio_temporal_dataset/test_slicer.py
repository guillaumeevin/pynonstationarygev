import pandas as pd
import numpy as np
from rpy2.rinterface import RRuntimeError
import unittest
from itertools import product

from extreme_estimator.extreme_models.margin_model.smooth_margin_model import ConstantMarginModel
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Smith
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.unidimensional_coordinates.coordinates_1D import LinSpaceCoordinates
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset, FullSimulatedDataset
from spatio_temporal_dataset.slicer.spatial_slicer import SpatialSlicer
from spatio_temporal_dataset.slicer.spatio_temporal_slicer import SpatioTemporalSlicer
from spatio_temporal_dataset.slicer.split import ALL_SPLITS_EXCEPT_ALL, Split
from spatio_temporal_dataset.slicer.temporal_slicer import TemporalSlicer
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestSlicerForDataset(unittest.TestCase):

    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.dataset = None

    nb_spatial_points = 2
    nb_temporal_obs = 2
    complete_shape = (nb_spatial_points, nb_temporal_obs)

    def load_dataset(self, slicer_class, split_ratio_spatial, split_ratio_temporal):
        coordinates = LinSpaceCoordinates.from_nb_points(nb_points=self.nb_spatial_points,
                                                         train_split_ratio=split_ratio_spatial)
        return FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_temporal_obs,
                                                         train_split_ratio=split_ratio_temporal,
                                                         margin_model=ConstantMarginModel(coordinates=coordinates),
                                                         coordinates=coordinates, max_stable_model=Smith(),
                                                         slicer_class=slicer_class)

    def get_shape(self, dataset, split):
        return dataset.maxima_frech(split).shape

    def test_spatiotemporal_slicer_for_dataset(self):
        ind_tuple_to_observation_shape = {
            (None, None): self.complete_shape,
            (None, 0.5): self.complete_shape,
            (0.5, None): self.complete_shape,
            (0.5, 0.5): (1, 1),
        }
        self.check_shapes(ind_tuple_to_observation_shape, SpatioTemporalSlicer)

    def test_spatial_slicer_for_dataset(self):
        ind_tuple_to_observation_shape = {
            (None, None): self.complete_shape,
            (None, 0.5): self.complete_shape,
            (0.5, None): (1, 2),
            (0.5, 0.5): (1, 2),
        }
        self.check_shapes(ind_tuple_to_observation_shape, SpatialSlicer)

    def test_temporal_slicer_for_dataset(self):
        ind_tuple_to_observation_shape = {
            (None, None): self.complete_shape,
            (None, 0.5): (2, 1),
            (0.5, None): self.complete_shape,
            (0.5, 0.5): (2, 1),
        }
        self.check_shapes(ind_tuple_to_observation_shape, TemporalSlicer)

    def check_shapes(self, ind_tuple_to_observation_shape, slicer_type):
        for split_ratio, data_shape in ind_tuple_to_observation_shape.items():
            dataset = self.load_dataset(slicer_type, *split_ratio)
            self.assertEqual(self.complete_shape, self.get_shape(dataset, Split.all))
            for split in ALL_SPLITS_EXCEPT_ALL:
                if split in dataset.slicer.splits:
                    self.assertEqual(data_shape, self.get_shape(dataset, split))
                else:
                    with self.assertRaises(AssertionError):
                        self.get_shape(dataset, split)


class TestSlicerForCoordinates(unittest.TestCase):

    def nb_coordinates(self, coordinates: AbstractCoordinates, split):
        return len(coordinates.coordinates_values(split))

    def test_slicer_for_coordinates(self):
        for split in Split:
            coordinates1 = LinSpaceCoordinates.from_nb_points(nb_points=2, train_split_ratio=0.5)
            if split in SpatialSlicer.SPLITS:
                self.assertEqual(self.nb_coordinates(coordinates1, split), 1)
            elif split in SpatioTemporalSlicer.SPLITS:
                self.assertEqual(self.nb_coordinates(coordinates1, split), 1)
            elif split in TemporalSlicer.SPLITS:
                self.assertEqual(self.nb_coordinates(coordinates1, split), 2)
            else:
                self.assertEqual(self.nb_coordinates(coordinates1, split), 2)
            coordinates2 = LinSpaceCoordinates.from_nb_points(nb_points=2)
            self.assertEqual(self.nb_coordinates(coordinates2, split), 2)


class TestSlicerForObservations(unittest.TestCase):

    def load_observations(self, split_ratio_temporal):
        df = pd.DataFrame.from_dict(
            {
                'year1': [1 for _ in range(4)],
                'year2': [2 for _ in range(4)],

            })
        return AbstractSpatioTemporalObservations(df_maxima_frech=df, train_split_ratio=split_ratio_temporal)

    def nb_obs(self, observations, split, slicer):
        return len(np.transpose(observations.maxima_frech(split, slicer)))

    def test_slicer_for_observations(self):
        observations = self.load_observations(0.5)
        # For the None Slicer, a slice should be returned only for split=SpatialTemporalSplit.all
        # self.assertEqual(len(observations.maxima_frech(SpatialTemporalSplit.all, None)), 2)
        self.assertEqual(2, self.nb_obs(observations, Split.all, None))
        for split in ALL_SPLITS_EXCEPT_ALL:
            with self.assertRaises(AssertionError):
                observations.maxima_frech(split, None)
        # For other slicers we try out all the possible combinations
        slicer_type_to_size = {
            SpatialSlicer: 2,
            TemporalSlicer: 1,
            SpatioTemporalSlicer: 1,
        }
        for slicer_type, size in slicer_type_to_size.items():
            for coordinates_train_ind in [None, pd.Series([True, True, True, False])][::-1]:
                slicer = slicer_type(coordinates_train_ind=coordinates_train_ind,
                                     observations_train_ind=observations.train_ind)
                self.assertEqual(2, self.nb_obs(observations, Split.all, slicer))
                for split in ALL_SPLITS_EXCEPT_ALL:
                    if split in slicer.splits:
                        # By default for SpatioTemporalSlicer should slice if both train_ind are available
                        # Otherwise if coordinates_train_ind is None, then it should return all the data
                        if slicer_type is SpatioTemporalSlicer and coordinates_train_ind is None:
                            size = 2
                        self.assertEqual(size, self.nb_obs(observations, split, slicer))
                    else:
                        with self.assertRaises(AssertionError):
                            observations.maxima_frech(split, slicer)


if __name__ == '__main__':
    unittest.main()
