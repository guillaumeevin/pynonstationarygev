import unittest

from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from extreme_estimator.extreme_models.max_stable_model.utils import load_max_stable_models
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from spatio_temporal_dataset.coordinates.spatial_coordinates.generated_spatial_coordinates import CircleCoordinates


class TestMaxStableEstimators(unittest.TestCase):
    DISPLAY = False

    MAX_STABLE_ESTIMATORS = [MaxStableEstimator]

    def setUp(self):
        super().setUp()
        self.spatial_coord = CircleCoordinates.from_nb_points(nb_points=5, max_radius=1)
        self.max_stable_models = load_max_stable_models()

    def test_max_stable_estimators(self):
        for max_stable_model in self.max_stable_models:
            dataset = MaxStableDataset.from_sampling(nb_obs=10,
                                                     max_stable_model=max_stable_model,
                                                     coordinates=self.spatial_coord)

            for estimator_class in self.MAX_STABLE_ESTIMATORS:
                estimator = estimator_class(dataset=dataset, max_stable_model=max_stable_model)
                estimator.fit()
                if self.DISPLAY:
                    print(type(max_stable_model))
                    print(dataset.df_dataset.head())
                    print(estimator.additional_information)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
