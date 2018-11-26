import unittest

from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates
from test.test_utils import load_test_max_stable_models, load_test_1D_and_2D_coordinates, \
    load_test_max_stable_estimators


class TestMaxStableEstimators(unittest.TestCase):
    DISPLAY = False
    nb_points = 5
    nb_obs = 10

    def setUp(self):
        super().setUp()
        self.coordinates = load_test_1D_and_2D_coordinates(nb_points=self.nb_points)
        self.max_stable_models = load_test_max_stable_models()

    def test_max_stable_estimators(self):
        for coordinates in self.coordinates:
            for max_stable_model in self.max_stable_models:
                dataset = MaxStableDataset.from_sampling(nb_obs=self.nb_obs,
                                                         max_stable_model=max_stable_model,
                                                         coordinates=coordinates)

                for max_stable_estimator in load_test_max_stable_estimators(dataset, max_stable_model):
                    max_stable_estimator.fit()
                    if self.DISPLAY:
                        print(type(max_stable_model))
                        print(dataset.df_dataset.head())
                        print(max_stable_estimator.additional_information)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
