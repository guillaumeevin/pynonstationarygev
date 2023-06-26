import unittest
from itertools import product

import numpy as np

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import \
    LinearNonStationaryLocationMarginModel
from extreme_fit.model.utils import set_seed_for_test
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset, MarginDataset
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import AnnualMaxima
from test.test_utils import load_test_max_stable_models, \
    load_test_1D_and_2D_spatial_coordinates, load_test_spatiotemporal_coordinates


class TestDataset(unittest.TestCase):
    nb_obs = 2
    nb_points = 2

    def test_remove_zero_from_dataset(self):
        coordinates, dataset_initial, observations = self.build_initial_dataset()
        dataset_without_zero = AbstractDataset.remove_zeros(observations,
                                                            coordinates)
        self.assertEqual(len(dataset_initial.coordinates), 4)
        self.assertEqual(len(dataset_without_zero.coordinates), 2)

    def test_remove_top_maxima_from_dataset(self):
        coordinates, dataset_initial, observations = self.build_initial_dataset()
        dataset_without_top = AbstractDataset.remove_top_maxima(observations,
                                                                coordinates)
        self.assertEqual(4, len(dataset_initial.coordinates), 4)
        self.assertEqual(2, len(dataset_without_top.coordinates))
        maxima = list(dataset_without_top.observations.df_maxima_gev.values[:, 0])
        self.assertEqual(set(maxima), {0})

    def test_remove_last_maxima_from_dataset(self):
        coordinates, dataset_initial, observations = self.build_initial_dataset()
        dataset_without_last = AbstractDataset.remove_last_maxima(observations,
                                                                  coordinates,
                                                                  nb_years=1)
        self.assertEqual(4, len(dataset_initial.coordinates), 4)
        self.assertEqual(2, len(dataset_without_last.coordinates))
        maxima = list(dataset_without_last.observations.df_maxima_gev.values[:, 0])
        self.assertEqual(set(maxima), {0, 2})

    def build_initial_dataset(self):
        temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=2)
        spatial_coordinates = AbstractSpatialCoordinates.from_list_x_coordinates([300, 600])
        coordinates = AbstractSpatioTemporalCoordinates(spatial_coordinates=spatial_coordinates,
                                                        temporal_coordinates=temporal_coordinates)
        coordinate_values_to_maxima = {
            (300, 0): [0],
            (300, 1): [1],
            (600, 0): [2],
            (600, 1): [0],
        }
        observations = AnnualMaxima.from_coordinates(coordinates, coordinate_values_to_maxima)
        dataset_initial = AbstractDataset(observations, coordinates)
        return coordinates, dataset_initial, observations

    def test_max_stable_dataset_R1_and_R2(self):
        max_stable_models = load_test_max_stable_models()[:]
        coordinates = load_test_1D_and_2D_spatial_coordinates(self.nb_points)
        for coordinates, max_stable_model in product(coordinates, max_stable_models):
            dataset = MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                                     max_stable_model=max_stable_model,
                                                     coordinates=coordinates)
            assert len(dataset.df_dataset.columns) == self.nb_obs + dataset.coordinates.nb_coordinates
        self.assertTrue(True)


class TestSpatioTemporalDataset(unittest.TestCase):
    nb_obs = 2
    nb_points = 3
    nb_steps = 2

    def setUp(self) -> None:
        set_seed_for_test(seed=42)
        self.coordinates = load_test_spatiotemporal_coordinates(nb_steps=self.nb_steps, nb_points=self.nb_points)[1]

    def load_dataset(self, nb_obs):
        smooth_margin_model = LinearNonStationaryLocationMarginModel(coordinates=self.coordinates,
                                                                     starting_point=1)
        self.dataset = MarginDataset.from_sampling(nb_obs=nb_obs,
                                                   margin_model=smooth_margin_model,
                                                   coordinates=self.coordinates)
        print(self.dataset.__str__())

    def test_spatio_temporal_array_wrt_time(self):
        # The test could have been on a given station. But we decided to do it for a given time step.
        self.load_dataset(nb_obs=1)

        # Load observation for time 0
        ind_time_0 = self.dataset.coordinates.ind_of_df_all_coordinates(
            coordinate_name=AbstractCoordinates.COORDINATE_T,
            value=0)
        observation_at_time_0_v1 = self.dataset.observations.df_maxima_gev.loc[ind_time_0].values.flatten()

        # Load observation correspond to time 0
        maxima_gev = self.dataset.maxima_gev_for_spatial_extremes_package
        maxima_gev = np.transpose(maxima_gev)
        self.assertEqual(maxima_gev.shape, (3, 2))
        observation_at_time_0_v2 = maxima_gev[:, 0]
        equality = np.equal(observation_at_time_0_v1, observation_at_time_0_v2).all()
        self.assertTrue(equality, msg='v1={} is different from v2={}'.format(observation_at_time_0_v1,
                                                                             observation_at_time_0_v2))

    def test_spatio_temporal_array_wrt_space(self):
        # The test could have been on a given station. But we decided to do it for a given time step.
        self.load_dataset(nb_obs=1)

        # Load observation for time 0
        ind_station_0 = self.dataset.coordinates.ind_of_df_all_coordinates(
            coordinate_name=AbstractCoordinates.COORDINATE_X,
            value=-1)
        observation_at_station_0_v1 = self.dataset.observations.df_maxima_gev.loc[ind_station_0].values.flatten()

        # Load observation correspond to time 0
        maxima_gev = self.dataset.maxima_gev_for_spatial_extremes_package
        maxima_gev = np.transpose(maxima_gev)
        self.assertEqual(maxima_gev.shape, (3, 2))
        observation_at_time_0_v2 = maxima_gev[0, :]
        equality = np.equal(observation_at_station_0_v1, observation_at_time_0_v2).all()
        self.assertTrue(equality, msg='v1={} is different from v2={}'.format(observation_at_station_0_v1,
                                                                             observation_at_time_0_v2))

    def test_spatio_temporal_array_with_multiple_observations(self):
        # In this case, we must check that the observations are the same
        self.load_dataset(nb_obs=2)

        # Load observation for time 0
        ind_station_0 = self.dataset.coordinates.ind_of_df_all_coordinates(
            coordinate_name=AbstractCoordinates.COORDINATE_X,
            value=-1)
        observation_at_station_0_v1 = self.dataset.observations.df_maxima_gev.loc[ind_station_0].values.flatten()
        # Load observation correspond to time 0
        maxima_gev = self.dataset.maxima_gev_for_spatial_extremes_package
        maxima_gev = np.transpose(maxima_gev)
        self.assertEqual(maxima_gev.shape, (3, 2 * 2))
        observation_at_station_0_v2 = maxima_gev[0, :]
        self.assertEqual(len(observation_at_station_0_v2), 4, msg='{}'.format(observation_at_station_0_v2))

        # The order does not really matter here but we check it anyway
        self.assertTrue(np.equal(observation_at_station_0_v1, observation_at_station_0_v2).all(),
                        msg='v1={} is different from v2={}'.format(observation_at_station_0_v1,
                                                                   observation_at_station_0_v2))


if __name__ == '__main__':
    unittest.main()
