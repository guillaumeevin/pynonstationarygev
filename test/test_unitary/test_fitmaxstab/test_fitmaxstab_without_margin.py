import unittest

from extreme_estimator.estimator.max_stable_estimator.abstract_max_stable_estimator import MaxStableEstimator
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Schlather
from extreme_estimator.extreme_models.utils import r
from spatio_temporal_dataset.dataset.simulation_dataset import MaxStableDataset
from test.test_unitary.test_rmaxstab.test_rmaxstab_without_margin import TestRMaxStab
from test.test_unitary.test_unitary_abstract import TestUnitaryAbstract


class TestMaxStableFitWithoutMargin(TestUnitaryAbstract):

    @property
    def r_output(self):
        TestRMaxStab.r_code()
        r("""res = fitmaxstab(data, locations, "whitmat")""")
        return self.r_fitted_values_from_res_variable

    @property
    def python_output(self):
        coordinates, max_stable_model = TestRMaxStab.python_code()
        dataset = MaxStableDataset.from_sampling(nb_obs=40, max_stable_model=max_stable_model, coordinates=coordinates)
        max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, use_start_value=False)
        max_stable_estimator = MaxStableEstimator(dataset, max_stable_model)
        max_stable_estimator.fit()
        return max_stable_estimator.max_stable_params_fitted

    def test_max_stable_fit_without_margin(self):
        self.compare()


if __name__ == '__main__':
    unittest.main()
