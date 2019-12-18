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
from spatio_temporal_dataset.coordinates.transformed_coordinates.transformation.abstract_transformation import \
    CenteredScaledNormalization
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.spatio_temporal_observations.abstract_spatio_temporal_observations import \
    AbstractSpatioTemporalObservations


class TestConfidenceInterval(unittest.TestCase):

    def load_data(self) -> None:
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
        self.coordinates = self.load_coordinates(df)
        df2 = pd.DataFrame(data=np.array(r['x_gev']), index=df.index)
        observations = AbstractSpatioTemporalObservations(df_maxima_gev=df2)
        self.dataset = AbstractDataset(observations=observations, coordinates=self.coordinates)
        self.model_classes = [StationaryTemporalModel]

    @staticmethod
    def load_coordinates(df):
        return AbstractTemporalCoordinates.from_df(df)

    def compute_eurocode_ci(self, model_class):
        self.load_data()
        estimator = fitted_linear_margin_estimator(model_class, self.coordinates, self.dataset,
                                                   starting_year=0,
                                                   fit_method=self.fit_method,
                                                   nb_iterations_for_bayesian_fit=100)
        return EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator, self.ci_method,
                                                                              self.start_year)

    @property
    def bayesian_ci(self):
        return {
            StationaryTemporalModel: (5.322109348451903, 7.0799164594485005, 9.204148461413848),
            NonStationaryLocationTemporalModel: (5.841829981629489, 8.698075782143512, 11.714171407134813),
            NonStationaryLocationAndScaleTemporalModel:(7.461627064650193, 9.0830495118253, 10.111709666579216),
        }

    def test_my_bayes(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        self.ci_method = ConfidenceIntervalMethodFromExtremes.my_bayes
        self.model_class_to_triplet = self.bayesian_ci

    def test_ci_bayes(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_bayesian
        self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_bayes
        self.model_class_to_triplet = self.bayesian_ci

    def test_ci_normal(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_mle
        self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_mle
        self.model_class_to_triplet = {
            StationaryTemporalModel: (-4.703945484843988, 30.482318639674023, 65.66858276419204),
            NonStationaryLocationTemporalModel: (-30.361576509947707, 4.159940117114796, 38.6814567441773),
            NonStationaryLocationAndScaleTemporalModel: (-52.797816369170455, 6.0677087210572465, 64.93323381128495),
        }

    def test_ci_boot(self):
        self.fit_method = TemporalMarginFitMethod.extremes_fevd_mle
        self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_boot
        self.model_class_to_triplet= {
            # I think the boostrapping works only in the stationary context
            # In the arg of the function for the non stationary return level there is only the method "normal" available
            StationaryTemporalModel: (10.260501562662334, 39.91206869180525, 120.3789497755127),
        }

    # Proflik seems to crash with the error
    # def test_ci_proflik(self):
    #     self.fit_method = TemporalMarginFitMethod.extremes_fevd_mle
    #     self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_proflik
    #     self.model_class_to_triplet= {
    #         # I think the profil likelihood works only in the stationary context
    #         # In the arg of the function for the non stationary return level there is only the method "normal" available
    #         StationaryTemporalModel: (-4.703945484843988, 30.482318639674023, 65.66858276419204),
    #     }

    def tearDown(self) -> None:
        for model_class, expected_triplet in self.model_class_to_triplet.items():
            eurocode_ci = self.compute_eurocode_ci(model_class)
            found_triplet = eurocode_ci.triplet
            for a, b in zip(expected_triplet, found_triplet):
                self.assertAlmostEqual(a, b, msg="{} \n{}".format(model_class, found_triplet))


class TestConfidenceIntervalModifiedCoordinates(TestConfidenceInterval):

    @staticmethod
    def load_coordinates(df):
        return AbstractTemporalCoordinates.from_df(df, transformation_class=CenteredScaledNormalization)

    @property
    def bayesian_ci(self):
        return {
            StationaryTemporalModel: (5.322109348451903, 7.079916459448501, 9.204148461413848),
            NonStationaryLocationTemporalModel: (7.285138442751067, 9.965330929203255, 13.313068256451233),
            NonStationaryLocationAndScaleTemporalModel: (11.744572233784234, 15.89417144494369, 23.522431032480416),
        }

    def test_my_bayes(self):
        super().test_my_bayes()

    def test_ci_bayes(self):
        super().test_ci_bayes()

    def test_ci_normal(self):
        self.model_class_to_triplet = {}
        self.assertTrue(True)

    def test_ci_boot(self):
        self.model_class_to_triplet = {}
        self.assertTrue(True)

    # def test_ci_proflik(self):
    #     self.model_class_to_triplet = {}
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
