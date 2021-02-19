import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocationLinear, AltitudinalShapeConstantTimeScaleLinear, \
    AltitudinalShapeConstantTimeLocScaleLinear
from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.altitude_spatial_model.altitudes_fit.altitudes_studies import AltitudesStudies
from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.one_fold_fit import OneFoldFit
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate, AnomalyTemperatureTemporalCovariate


class TestOneFoldFit(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.altitudes = [1200, 1500, 1800]
        self.massif_name = "Vanoise"
        self.model_classes = [StationaryAltitudinal,
                              AltitudinalShapeConstantTimeLocationLinear,
                              AltitudinalShapeConstantTimeScaleLinear,
                              AltitudinalShapeConstantTimeLocScaleLinear
                              ][:]

    def load_dataset(self, study_class, **kwargs_study):
        self.studies = AltitudesStudies(study_class, self.altitudes, **kwargs_study)
        dataset = self.studies.spatio_temporal_dataset(massif_name=self.massif_name)
        return dataset

    def test_without_temporal_covariate(self):
        for study_class in [SafranSnowfall1Day, AdamontSnowfall][:]:
            dataset = self.load_dataset(study_class)
            one_fold_fit = OneFoldFit(self.massif_name, dataset,
                                      models_classes=self.model_classes, temporal_covariate_for_fit=None,
                                      only_models_that_pass_goodness_of_fit_test=False)
            _ = one_fold_fit.best_estimator.margin_model
        self.assertTrue(True)

    def test_with_temporal_covariate_for_time(self):
        for study_class in [SafranSnowfall1Day, AdamontSnowfall][:]:
            dataset = self.load_dataset(study_class)
            one_fold_fit = OneFoldFit(self.massif_name, dataset,
                                      models_classes=self.model_classes,
                                      temporal_covariate_for_fit=TimeTemporalCovariate,
                                      only_models_that_pass_goodness_of_fit_test=False)
            _ = one_fold_fit.best_estimator.margin_model
        self.assertTrue(True)

    def test_with_temporal_covariate_for_temperature_anomaly(self):
        for study_class in [AdamontSnowfall][:]:
            dataset = self.load_dataset(study_class, scenario=AdamontScenario.rcp85)
            one_fold_fit = OneFoldFit(self.massif_name, dataset,
                                      models_classes=self.model_classes,
                                      temporal_covariate_for_fit=AnomalyTemperatureTemporalCovariate,
                                      only_models_that_pass_goodness_of_fit_test=False)
            _ = one_fold_fit.best_estimator.margin_model
        self.assertTrue(True)

    def test_remove_physically_implausible_models(self):
        self.massif_name = "Aravis"
        self.altitudes = [600, 900]
        self.model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
        dataset = self.load_dataset(AdamontSnowfall,
                                    scenario=AdamontScenario.rcp85, gcm_rcm_couple=('CNRM-CM5', 'CCLM4-8-17'))
        one_fold_fit = OneFoldFit(self.massif_name, dataset,
                                  models_classes=self.model_classes,
                                  temporal_covariate_for_fit=None,
                                  only_models_that_pass_goodness_of_fit_test=False,
                                  remove_physically_implausible_models=True)
        self.assertFalse(one_fold_fit.has_at_least_one_valid_model)


if __name__ == '__main__':
    unittest.main()
