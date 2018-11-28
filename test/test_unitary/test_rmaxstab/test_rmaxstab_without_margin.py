import unittest

import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import CovarianceFunction
from extreme_estimator.extreme_models.max_stable_model.max_stable_models import Schlather
from extreme_estimator.extreme_models.utils import r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.temporal_observations.annual_maxima_observations import MaxStableAnnualMaxima
from test.test_unitary.test_unitary_abstract import TestUnitaryAbstract


class TestRMaxStab(TestUnitaryAbstract):

    @classmethod
    def r_code(cls):
        r("""data <- rmaxstab(40, locations, cov.mod = "whitmat", nugget = 0, range = 3, smooth = 0.5)""")

    @classmethod
    def python_code(cls):
        # Load coordinate object
        df = pd.DataFrame(data=r.locations, columns=AbstractCoordinates.COORDINATE_NAMES[:2])
        coordinates = AbstractCoordinates.from_df(df)
        # Load max stable model
        params_sample = {'range': 3, 'smooth': 0.5, 'nugget': 0}
        max_stable_model = Schlather(covariance_function=CovarianceFunction.whitmat, params_sample=params_sample)
        return coordinates, max_stable_model

    @property
    def r_output(self):
        self.r_code()
        return np.sum(r.data)

    @property
    def python_output(self):
        coordinates, max_stable_model = self.python_code()
        m = MaxStableAnnualMaxima.from_sampling(nb_obs=40, max_stable_model=max_stable_model, coordinates=coordinates)
        # TODO: understand why the array are not in the same order
        return np.sum(m.maxima_frech)

    def test_rmaxstab(self):
        self.compare()


if __name__ == '__main__':
    unittest.main()
