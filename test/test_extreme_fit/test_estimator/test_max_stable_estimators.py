import unittest

from extreme_fit.model.utils import SafeRunException
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from test.test_utils import load_test_max_stable_models, load_test_1D_and_2D_spatial_coordinates, \
    load_test_max_stable_estimators


class TestMaxStableEstimators(unittest.TestCase):
    DISPLAY = False
    nb_points = 2
    nb_obs = 3

    def setUp(self):
        super().setUp()
        self.coordinates = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points)

        self.max_stable_models = load_test_max_stable_models()

    def test_max_stable_estimators(self):
        self.fit_max_stable_estimator_for_all_coordinates()
        self.assertTrue(True)

    def fit_max_stable_estimator_for_all_coordinates(self):
        for coordinates in self.coordinates:
            for max_stable_model in self.max_stable_models:
                use_rmaxstab_with_2_coordinates = coordinates.nb_spatial_coordinates > 2
                dataset = MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                                         max_stable_model=max_stable_model,
                                                         coordinates=coordinates,
                                                         use_rmaxstab_with_2_coordinates=use_rmaxstab_with_2_coordinates)
                for max_stable_estimator in load_test_max_stable_estimators(dataset, max_stable_model):
                    max_stable_estimator.fit()
                    if self.DISPLAY:
                        print(type(max_stable_model))
                        print(dataset.df_dataset.head())
                        print(max_stable_estimator.additional_information)



if __name__ == '__main__':
    unittest.main()
