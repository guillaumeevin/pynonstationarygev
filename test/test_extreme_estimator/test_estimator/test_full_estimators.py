import unittest
from itertools import product

from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from test.test_utils import load_test_max_stable_models, load_smooth_margin_models, load_test_1D_and_2D_spatial_coordinates, \
    load_test_full_estimators


class TestFullEstimators(unittest.TestCase):
    DISPLAY = False
    nb_obs = 3
    nb_points = 2

    def setUp(self):
        super().setUp()
        self.spatial_coordinates = load_test_1D_and_2D_spatial_coordinates(nb_points=self.nb_points)
        self.max_stable_models = load_test_max_stable_models()

    def test_full_estimators(self):
        for coordinates in self.spatial_coordinates:
            smooth_margin_models = load_smooth_margin_models(coordinates=coordinates)
            for margin_model, max_stable_model in product(smooth_margin_models, self.max_stable_models):
                dataset = FullSimulatedDataset.from_double_sampling(nb_obs=self.nb_obs, margin_model=margin_model,
                                                                    coordinates=coordinates,
                                                                    max_stable_model=max_stable_model)

                for full_estimator in load_test_full_estimators(dataset, margin_model, max_stable_model):
                    full_estimator.fit()
                    if self.DISPLAY:
                        print(type(margin_model))
                        print(dataset.df_dataset.head())
                        print(full_estimator.additional_information)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
