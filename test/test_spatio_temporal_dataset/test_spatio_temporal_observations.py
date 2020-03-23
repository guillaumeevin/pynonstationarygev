import unittest

import numpy as np
import pandas as pd

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.utils import set_seed_for_test
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations
from spatio_temporal_dataset.spatio_temporal_observations.annual_maxima_observations import DailyExpAnnualMaxima
from spatio_temporal_dataset.spatio_temporal_observations.daily_observations import DailyExp


class TestSpatioTemporalObservations(unittest.TestCase):
    DISPLAY = False

    def test_set_maxima_gev(self):
        df = pd.DataFrame.from_dict({'ok': [2, 5]})
        temporal_observation = AbstractSpatioTemporalObservations(df_maxima_frech=df)
        example = np.array([[3], [6]])
        temporal_observation.set_maxima_frech(maxima_frech_values=example)
        maxima_frech = temporal_observation.maxima_frech()
        self.assertTrue(np.equal(example, maxima_frech).all(), msg="{} {}".format(example, maxima_frech))


class TestDailyObservations(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        set_seed_for_test(seed=42)
        self.coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_temporal_steps=10)
        gev_param_name_to_coef_list = {
            AbstractParams.RATE: [1],
        }
        self.margin_model = StationaryTemporalModel.from_coef_list(self.coordinates, gev_param_name_to_coef_list,
                                                                   params_class=ExpParams)

    def test_instance_exp_params(self):
        last_coordinate = self.coordinates.coordinates_values()[-1]
        params = self.margin_model.margin_function_sample.get_gev_params(last_coordinate)
        self.assertIsInstance(params, ExpParams)

    def test_exponential_observations(self):
        obs = DailyExp.from_sampling(nb_obs=1, coordinates=self.coordinates,
                                     margin_model=self.margin_model)
        self.assertAlmostEqual(obs.df_maxima_gev.mean()[0], 0.574829320536985)

    def test_annual_maxima_observations_from_daily_observations(self):
        obs = DailyExpAnnualMaxima.from_sampling(1, self.coordinates, self.margin_model)
        self.assertAlmostEqual(obs.df_maxima_gev.mean()[0], 6.523848726794694)


if __name__ == '__main__':
    unittest.main()
