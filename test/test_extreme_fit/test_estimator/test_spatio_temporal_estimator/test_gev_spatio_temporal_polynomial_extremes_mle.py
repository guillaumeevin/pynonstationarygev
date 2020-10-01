import unittest

import numpy as np
import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.utils import fitted_linear_margin_estimator_short
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocationLinear
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeLocationLinear, AltitudinalShapeLinearTimeLocScaleLinear, \
    AltitudinalShapeLinearTimeStationary
from extreme_fit.model.margin_model.polynomial_margin_model.polynomial_margin_model import \
    NonStationaryQuadraticLocationModel, \
    NonStationaryQuadraticScaleModel, NonStationaryQuadraticLocationGumbelModel, NonStationaryQuadraticScaleGumbelModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from extreme_trend.abstract_gev_trend_test import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from extreme_fit.model.utils import r, set_seed_r
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.spatial_coordinates.abstract_spatial_coordinates import \
    AbstractSpatialCoordinates
from spatio_temporal_dataset.coordinates.spatio_temporal_coordinates.abstract_spatio_temporal_coordinates import \
    AbstractSpatioTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.generated_temporal_coordinates import \
    ConsecutiveTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestGevTemporalQuadraticExtremesMle(unittest.TestCase):

    def setUp(self) -> None:
        nb_data = 100
        set_seed_r()
        r("""
        N <- {}
        loc = 0; scale = 1; shape <- 0.1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """.format(nb_data))

        # Compute coordinates
        altitudes = [300, 600]
        temporal_coordinates = ConsecutiveTemporalCoordinates.from_nb_temporal_steps(nb_data)
        spatial_coordinates = AbstractSpatialCoordinates.from_list_x_coordinates(altitudes)
        self.coordinates = AbstractSpatioTemporalCoordinates.from_spatial_coordinates_and_temporal_coordinates(
            spatial_coordinates,
            temporal_coordinates)

        # Compute observations
        a = np.array(r['x_gev'])
        data = np.concatenate([a, a], axis=0)
        df2 = pd.DataFrame(data=data, index=self.coordinates.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)

        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        self.fit_method = MarginFitMethod.extremes_fevd_mle

    def function_test_gev_spatio_temporal_margin_fit_non_stationary(self, model_class):
        # Create estimator
        estimator = fitted_linear_margin_estimator_short(model_class=model_class,
                                                         dataset=self.dataset,
                                                         fit_method=self.fit_method)

        # Assert that indicators are correctly computed
        self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh())
        self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic())
        self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic())
        # Assert we can compute the return level
        covariate1_for_return_level = np.array([700, 0])
        covariate2_for_return_level = np.array([700, 1000])
        covariate3_for_return_level = np.array([400, 0])
        coordinates = [covariate1_for_return_level, covariate2_for_return_level, covariate3_for_return_level]
        for coordinate in coordinates:
            EurocodeConfidenceIntervalFromExtremes.quantile_level = 0.98
            confidence_interval = EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator,
                                                                                                 ci_method=ConfidenceIntervalMethodFromExtremes.ci_mle,
                                                                                                 temporal_covariate=coordinate)
            gev_params = estimator.function_from_fit.get_params(coordinate)
            # print(gev_params, coordinate)
            return_level = gev_params.return_level(return_period=50)
            # print("my return level", return_level)
            # print("their return level", confidence_interval.mean_estimate)
            # print("diff", confidence_interval.mean_estimate - return_level)
            # print('\n\n')
            self.assertAlmostEqual(return_level, confidence_interval.mean_estimate)
            self.assertFalse(np.isnan(confidence_interval.confidence_interval[0]))
            self.assertFalse(np.isnan(confidence_interval.confidence_interval[1]))

    def test_gev_spatio_temporal_margin_fit_1(self):
        self.function_test_gev_spatio_temporal_margin_fit_non_stationary(StationaryAltitudinal)

    #
    def test_gev_spatio_temporal_margin_fit_1_bis(self):
        self.function_test_gev_spatio_temporal_margin_fit_non_stationary(AltitudinalShapeLinearTimeStationary)

    def test_gev_spatio_temporal_margin_fit_2(self):
        # first model with both a time and altitude non stationarity
        self.function_test_gev_spatio_temporal_margin_fit_non_stationary(
            AltitudinalShapeConstantTimeLocationLinear)

    # def test_gev_spatio_temporal_margin_fit_3(self):
    #     self.function_test_gev_spatio_temporal_margin_fit_non_stationary(
    #         AltitudinalShapeLinearTimeLocationLinear)
    #
    # def test_gev_spatio_temporal_margin_fit_4(self):
    #     self.function_test_gev_spatio_temporal_margin_fit_non_stationary(
    #         AltitudinalShapeLinearTimeLocScaleLinear)


if __name__ == '__main__':
    unittest.main()
