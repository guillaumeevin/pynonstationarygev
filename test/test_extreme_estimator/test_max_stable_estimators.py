import unittest

from extreme_estimator.R_fit.max_stable_fit.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather
from extreme_estimator.estimator.max_stable_estimator import MaxStableEstimator
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestMaxStableEstimators(unittest.TestCase):
    DISPLAY = False
    MAX_STABLE_TYPES = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]
    MAX_STABLE_ESTIMATORS = [MaxStableEstimator]

    def setUp(self):
        super().setUp()
        self.spatial_coord = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.max_stable_models = self.load_max_stable_models()

    @classmethod
    def load_max_stable_models(cls):
        # Load all max stable model
        max_stable_models = []
        for max_stable_class in cls.MAX_STABLE_TYPES:
            if issubclass(max_stable_class, AbstractMaxStableModelWithCovarianceFunction):
                max_stable_models.extend([max_stable_class(covariance_function=covariance_function)
                                               for covariance_function in CovarianceFunction])
            else:
                max_stable_models.append(max_stable_class())
        return max_stable_models

    def test_max_stable_estimators(self):
        for max_stable_model in self.max_stable_models:
            dataset = MaxStableDataset.from_sampling(nb_obs=10,
                                                     max_stable_model=max_stable_model,
                                                     spatial_coordinates=self.spatial_coord)

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
