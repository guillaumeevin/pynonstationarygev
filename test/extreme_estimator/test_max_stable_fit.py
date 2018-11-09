import unittest

from extreme_estimator.R_fit.max_stable_fit.abstract_max_stable_model import \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import Smith, BrownResnick, Schlather, \
    Geometric, ExtremalT, ISchlather
from spatio_temporal_dataset.dataset.simulation_dataset import SimulatedDataset
from spatio_temporal_dataset.spatial_coordinates.generated_coordinates import CircleCoordinatesRadius1


class TestMaxStableFit(unittest.TestCase):
    MAX_STABLE_CLASSES = [Smith, BrownResnick, Schlather, Geometric, ExtremalT, ISchlather]

    def setUp(self):
        super().setUp()
        self.spatial_coord = CircleCoordinatesRadius1.from_nb_points(nb_points=5, max_radius=1)
        self.max_stable_models = []
        for max_stable_class in self.MAX_STABLE_CLASSES:
            if issubclass(max_stable_class, AbstractMaxStableModelWithCovarianceFunction):
                self.max_stable_models.extend([max_stable_class(covariance_function=covariance_function)
                                               for covariance_function in CovarianceFunction])
            else:
                self.max_stable_models.append(max_stable_class())

    def test_sampling_fit_with_models(self, display=False):
        for max_stable_model in self.max_stable_models:
            dataset = SimulatedDataset.from_max_stable_sampling(nb_obs=10, max_stable_model=max_stable_model,
                                                                spatial_coordinates=self.spatial_coord)
            fitted_values = max_stable_model.fitmaxstab(maxima=dataset.maxima, coord=dataset.coord)
            if display:
                print(type(max_stable_model))
                print(dataset.df_dataset.head())
                print(fitted_values)
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
