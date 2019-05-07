import unittest

import numpy as np
import pandas as pd

from extreme_estimator.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_estimator.extreme_models.margin_model.temporal_linear_margin_model import StationaryStationModel, \
    NonStationaryStationModel
from extreme_estimator.extreme_models.utils import r, set_seed_r
from extreme_estimator.margin_fits.gev.gevmle_fit import GevMleFit
from extreme_estimator.margin_fits.gev.ismev_gev_fit import IsmevGevFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalMargin(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_r()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)
        # Compute the stationary temporal margin with isMev
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: range(50)})
        self.coordinates = AbstractTemporalCoordinates.from_df(df)
        df2 = pd.DataFrame(data=np.array(r['x_gev']), index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)

    def test_gev_temporal_margin_fit_stationary(self):
        # Create estimator
        margin_model = StationaryStationModel(self.coordinates)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        ref = {'loc': 0.0219, 'scale': 1.0347, 'shape': 0.8295}
        for year in range(1, 3):
            mle_params_estimated = estimator.margin_function_fitted.get_gev_params(np.array([year])).to_dict()
            for key in ref.keys():
                self.assertAlmostEqual(ref[key], mle_params_estimated[key], places=3)

    def test_gev_temporal_margin_fit_nonstationary(self):
        # Create estimator
        margin_model = NonStationaryStationModel(self.coordinates)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        self.assertNotEqual(estimator.result_from_fit.margin_coef_dict['tempCoeffLoc1'], 0.0)
        # Checks that parameters returned are indeed different
        mle_params_estimated_year1 = estimator.margin_function_fitted.get_gev_params(np.array([1])).to_dict()
        mle_params_estimated_year3 = estimator.margin_function_fitted.get_gev_params(np.array([3])).to_dict()
        self.assertNotEqual(mle_params_estimated_year1, mle_params_estimated_year3)

    def test_gev_temporal_margin_fit_nonstationary_with_start_point(self):
        # Create estimator
        margin_model = NonStationaryStationModel(self.coordinates, starting_point=3)
        estimator = LinearMarginEstimator(self.dataset, margin_model)
        estimator.fit()
        self.assertNotEqual(estimator.result_from_fit.margin_coef_dict['tempCoeffLoc1'], 0.0)
        # Checks starting point parameter are well passed
        self.assertEqual(3, estimator.margin_function_fitted.starting_point)
        # Checks that parameters returned are indeed different
        mle_params_estimated_year1 = estimator.margin_function_fitted.get_gev_params(np.array([1])).to_dict()
        mle_params_estimated_year3 = estimator.margin_function_fitted.get_gev_params(np.array([3])).to_dict()
        self.assertEqual(mle_params_estimated_year1, mle_params_estimated_year3)
        mle_params_estimated_year5 = estimator.margin_function_fitted.get_gev_params(np.array([5])).to_dict()
        self.assertNotEqual(mle_params_estimated_year5, mle_params_estimated_year3)
    # todo: create same test with a starting value, to check if the starting value is taken into account in the margin_function_fitted


if __name__ == '__main__':
    unittest.main()
