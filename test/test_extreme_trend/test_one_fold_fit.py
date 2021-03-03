import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.temperature_to_year import get_year_min_and_year_max
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocationLinear, AltitudinalShapeConstantTimeScaleLinear, \
    AltitudinalShapeConstantTimeLocScaleLinear
from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_trend.one_fold_fit.altitude_group import VeyHighAltitudeGroup
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


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
                                      temporal_covariate_for_fit=AnomalyTemperatureWithSplineTemporalCovariate,
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

    def test_assertion_error_for_a_specific_case(self):
        self.massif_name = "Thabor"
        self.model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS[:]
        self.altitudes = [3000, 3300, 3600]
        gcm_rcm_couple = ('HadGEM2-ES', 'RegCM4-6')
        scenario = AdamontScenario.rcp85
        year_min, year_max = get_year_min_and_year_max(gcm_rcm_couple[0], scenario, left_limit=1.0,
                                                       right_limit=2.5, is_temperature_interval=True)
        dataset = self.load_dataset(AdamontSnowfall,
                                    scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
                                    year_min=year_min, year_max=year_max)
        one_fold_fit = OneFoldFit(self.massif_name, dataset,
                                  models_classes=self.model_classes,
                                  temporal_covariate_for_fit=AnomalyTemperatureWithSplineTemporalCovariate,
                                  altitude_class=VeyHighAltitudeGroup,
                                  only_models_that_pass_goodness_of_fit_test=False,
                                  remove_physically_implausible_models=True)
        self.assertTrue(one_fold_fit.has_at_least_one_valid_model)

if __name__ == '__main__':
    unittest.main()
