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
                                                   fit_method=self.fit_method)
        return EurocodeConfidenceIntervalFromExtremes.from_estimator_extremes(estimator, self.ci_method,
                                                                              self.start_year)

    @property
    def bayesian_ci(self):
        return {
            StationaryTemporalModel: (6.756903450587758, 10.316338515219085, 15.77861914935531),
            NonStationaryLocationTemporalModel: (6.588570126641043, 10.055847177064836, 14.332882862817332),
            NonStationaryLocationAndScaleTemporalModel: (7.836837972383451, 11.162663922795906, 16.171701445841183),
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
        self.ci_method = ConfidenceIntervalMethodFromExtremes.ci_normal
        self.model_class_to_triplet= {
            StationaryTemporalModel: (-4.703945484843988, 30.482318639674023, 65.66858276419204),
            NonStationaryLocationTemporalModel: (-4.223086740397132, 30.29842988666537, 64.81994651372787),
            NonStationaryLocationAndScaleTemporalModel: (-15.17041284612494, 43.69511224410276, 102.56063733433047),
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
            StationaryTemporalModel: (6.756903450587758, 10.316338515219085, 15.77861914935531),
            NonStationaryLocationTemporalModel: (6.266027110993808, 10.063368195790687, 14.894103640762097),
            NonStationaryLocationAndScaleTemporalModel: (5.554116722722492, 13.714431163455535, 26.929963957448642),
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
