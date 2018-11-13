import unittest
from itertools import product

from extreme_estimator.estimator.full_estimator import SmoothMarginalsThenUnitaryMsp
from spatio_temporal_dataset.dataset.simulation_dataset import FullSimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1
from test.test_extreme_estimator.test_margin_estimators import TestMarginEstimators
from test.test_extreme_estimator.test_max_stable_estimators import TestMaxStableEstimators


class TestFullEstimators(unittest.TestCase):
    DISPLAY = False
    FULL_ESTIMATORS = [SmoothMarginalsThenUnitaryMsp]

    def setUp(self):
        super().setUp()
        self.spatial_coord = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.max_stable_models = TestMaxStableEstimators.load_max_stable_models()
        self.margin_models = TestMarginEstimators.load_margin_models()

    def test_full_estimators(self):
        for margin_model, max_stable_model in product(self.margin_models, self.max_stable_models):
            dataset = FullSimulatedDataset.from_double_sampling(nb_obs=10, margin_model=margin_model,
                                                                spatial_coordinates=self.spatial_coord,
                                                                max_stable_model=max_stable_model)

            for estimator_class in self.FULL_ESTIMATORS:
                estimator = estimator_class(dataset=dataset, margin_model=margin_model,
                                            max_stable_model=max_stable_model)
                estimator.fit()
                if self.DISPLAY:
                    print(type(margin_model))
                    print(dataset.df_dataset.head())
                    print(estimator.additional_information)
            self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
