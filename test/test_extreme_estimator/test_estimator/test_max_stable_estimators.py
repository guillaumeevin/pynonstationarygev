import unittest

from extreme_estimator.extreme_models.utils import SafeRunException
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.uniform_normalization import \
    BetweenZeroAndOneNormalization
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from test.test_utils import load_test_max_stable_models, load_test_1D_and_2D_spatial_coordinates, \
    load_test_max_stable_estimators, load_test_3D_spatial_coordinates


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
                use_rmaxstab_with_2_coordinates = coordinates.nb_coordinates_spatial > 2
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


class TestMaxStableEstimatorWorkingFor3DCoordinates(TestMaxStableEstimators):

    def setUp(self):
        super().setUp()
        self.coordinates = load_test_3D_spatial_coordinates(nb_points=self.nb_points,
                                                            transformation_class=BetweenZeroAndOneNormalization)
        # Select only the max stable structure that work with 3D coordinates
        self.max_stable_models = load_test_max_stable_models()[1:]


class TestMaxStableEstimatorGaussFor3DCoordinates(TestMaxStableEstimators):
    """
    See the fhe function rmaxstab3Dimprovedgauss in my file max_stable_fit.R
    it returns the following error when we try to fit a 3D smipth process:

    Error in nplk(p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10],  :
    objet 'C_smithdsgnmat3d' introuvable
    Calls: source ... rmaxstab3Dimprovedgauss -> fitmaxstab -> smithform -> do.call -> nllh -> nplk
    Exécution arrêtée
    """

    def setUp(self):
        super().setUp()
        self.coordinates = load_test_3D_spatial_coordinates(nb_points=self.nb_points,
                                                            transformation_class=BetweenZeroAndOneNormalization)
        self.max_stable_models = load_test_max_stable_models()[:1]

    def test_max_stable_estimators(self):
        with self.assertRaises(SafeRunException):
            self.fit_max_stable_estimator_for_all_coordinates()


if __name__ == '__main__':
    unittest.main()
