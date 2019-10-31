import unittest

import numpy as np
import pandas as pd

from experiment.trend_analysis.univariate_test.utils import fitted_linear_margin_estimator
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.eurocode_return_level_uncertainties import \
    EurocodeConfidenceIntervalFromExtremes
from extreme_fit.model.utils import r, set_seed_r, set_seed_for_test
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_coordinates import \
    AbstractTemporalCoordinates
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestConfidenceInterval(unittest.TestCase):

    def setUp(self) -> None:
        set_seed_for_test()
        r("""
        N <- 50
        loc = 0; scale = 1; shape <- 1
        x_gev <- rgev(N, loc = loc, scale = scale, shape = shape)
        start_loc = 0; start_scale = 1; start_shape = 1
        """)
        # Compute the stationary temporal margin with isMev
        self.start_year = 0
        df = pd.DataFrame({AbstractCoordinates.COORDINATE_T: range(self.start_year, self.start_year + 50)})
        self.coordinates = AbstractTemporalCoordinates.from_df(df)
        df2 = pd.DataFrame(data=np.array(r['x_gev']), index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        self.model_classes = [StationaryTemporalModel]

    def compute_eurocode_ci(self, model_class):
        estimator = fitted_linear_margin_estimator(model_class, self.coordinates, self.dataset,
                                                   starting_year=0,
                                                   fit_method=self.fit_method)
        return EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator, self.ci_method)

    def test_my_bayes(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        self.ci_method = ConfidenceIntervalMethodFromExtremes.my_bayes
        self.model_class_to_triplet = {
            StationaryTemporalModel: (6.756903450587758, 10.316338515219085, 15.77861914935531),
            NonStationaryLocationTemporalModel: (6.047033481540427, 9.708540966532225, 14.74058551926604),
            NonStationaryLocationAndScaleTemporalModel: (6.383536224810785, 9.69120774797993, 13.917914357321615),
        }

    def test_ci_bayes(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_bayes
        self.model_class_to_triplet = {
            StationaryTemporalModel: (6.756903450587758, 10.316338515219085, 15.77861914935531),
            # NonStationaryLocationTemporalModel: (6.047033481540427, 9.708540966532225, 14.74058551926604),
            # NonStationaryLocationAndScaleTemporalModel: (6.383536224810785, 9.69120774797993, 13.917914357321615),
        }

    def tearDown(self) -> None:
        for model_class, expected_triplet in self.model_class_to_triplet.items():
            eurocode_ci = self.compute_eurocode_ci(StationaryTemporalModel)
            found_triplet = eurocode_ci.triplet
            for a, b in zip(expected_triplet, found_triplet):
                self.assertAlmostEqual(a, b, msg="{} {}".format(model_class, found_triplet))


if __name__ == '__main__':
    unittest.main()
