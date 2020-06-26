import unittest
from random import sample

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranPrecipitation1Day
from extreme_fit.model.margin_model.polynomial_margin_model.altitudinal_models import \
    NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation, \
    NonStationaryAltitudinalLocationQuadraticCrossTermForLocation
from extreme_fit.model.margin_model.polynomial_margin_model.utils import ALTITUDINAL_MODELS, \
    MODELS_THAT_SHOULD_RAISE_AN_ASSERTION_ERROR, VARIOUS_SPATIO_TEMPORAL_MODELS
from extreme_fit.model.margin_model.utils import \
    MarginFitMethod
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.two_fold_analysis.two_fold_datasets_generator import \
    TwoFoldDatasetsGenerator
from projects.altitude_spatial_model.altitudes_fit.two_fold_analysis.two_fold_fit import TwoFoldFit


class TestGevTemporalQuadraticExtremesMle(unittest.TestCase):

    def setUp(self) -> None:
        self.altitudes = [900, 1200]
        self.massif_name = 'Vercors'
        self.study_class = SafranSnowfall1Day

    def get_estimator_fitted(self, model_class):
        studies = AltitudesStudies(self.study_class, self.altitudes, year_max=2019)
        two_fold_datasets_generator = TwoFoldDatasetsGenerator(studies, nb_samples=1, massif_names=[self.massif_name])
        model_family_name_to_model_class = {'Non stationary': [model_class]}
        two_fold_fit = TwoFoldFit(two_fold_datasets_generator=two_fold_datasets_generator,
                                  model_family_name_to_model_classes=model_family_name_to_model_class,
                                  fit_method=MarginFitMethod.extremes_fevd_mle)
        massif_fit = two_fold_fit.massif_name_to_massif_fit[self.massif_name]
        sample_fit = massif_fit.sample_id_to_sample_fit[0]
        model_fit = sample_fit.model_class_to_model_fit[model_class]  # type: TwoFoldModelFit
        estimator = model_fit.estimator_fold_1
        return estimator

    def common_test(self, model_class):
        estimator = self.get_estimator_fitted(model_class)
        # Assert that indicators are correctly computed
        self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh(split=estimator.train_split))
        self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic(split=estimator.train_split))
        self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic(split=estimator.train_split))

    def test_assert_error(self):
        for model_class in MODELS_THAT_SHOULD_RAISE_AN_ASSERTION_ERROR:
            with self.assertRaises(AssertionError):
                self.common_test(model_class)

    def test_location_spatio_temporal_models(self):
        for model_class in VARIOUS_SPATIO_TEMPORAL_MODELS:
            self.common_test(model_class)

    def test_altitudinal_models(self):
        for model_class in ALTITUDINAL_MODELS:
            # print(model_class)
            self.common_test(model_class)


# class MyTest(unittest.TestCase):
#
#     def setUp(self) -> None:
#         self.study_class = SafranPrecipitation1Day
#         self.altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000]
#         self.massif_name = 'Aravis'
#
#     def get_estimator_fitted(self, model_class):
#         studies = AltitudesStudies(self.study_class, self.altitudes, year_max=2019)
#         two_fold_datasets_generator = TwoFoldDatasetsGenerator(studies, nb_samples=1, massif_names=[self.massif_name])
#         model_family_name_to_model_class = {'Non stationary': [model_class]}
#         two_fold_fit = TwoFoldFit(two_fold_datasets_generator=two_fold_datasets_generator,
#                                   model_family_name_to_model_classes=model_family_name_to_model_class,
#                                   fit_method=MarginFitMethod.extremes_fevd_mle)
#         massif_fit = two_fold_fit.massif_name_to_massif_fit[self.massif_name]
#         sample_fit = massif_fit.sample_id_to_sample_fit[0]
#         model_fit = sample_fit.model_class_to_model_fit[model_class]  # type: TwoFoldModelFit
#         estimator = model_fit.estimator_fold_1
#         return estimator
#
#     def common_test(self, model_class):
#         estimator = self.get_estimator_fitted(model_class)
#         # Assert that indicators are correctly computed
#         self.assertAlmostEqual(estimator.result_from_model_fit.nllh, estimator.nllh(split=estimator.train_split))
#         self.assertAlmostEqual(estimator.result_from_model_fit.aic, estimator.aic(split=estimator.train_split))
#         self.assertAlmostEqual(estimator.result_from_model_fit.bic, estimator.bic(split=estimator.train_split))
#
#     # def test_altitudinal_models(self):
#     #     for model_class in ALTITUDINAL_MODELS:
#     #         self.common_test(model_class)
#
#     def test_wrong(self):
#         self.common_test(NonStationaryAltitudinalLocationQuadraticCrossTermForLocation)
#         # self.common_test(NonStationaryAltitudinalLocationQuadraticScaleLinearCrossTermForLocation)


if __name__ == '__main__':
    unittest.main()
