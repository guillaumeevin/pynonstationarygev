import unittest

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_constant_shape_wrt_altitude import \
    AltitudinalShapeConstantTimeLocationLinear, AltitudinalShapeConstantTimeScaleLinear, \
    AltitudinalShapeConstantTimeLocScaleLinear
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate


class TestEnsembleFit(unittest.TestCase):

    def setUp(self) -> None:
        super().setUp()
        self.altitudes = [1200, 1500, 1800]
        self.massif_names = ["Vanoise"]
        study_class = AdamontSnowfall
        gcm_rcm_couples = [('CNRM-CM5', 'ALADIN53'), ('EC-EARTH', 'CCLM4-8-17')]
        scenario = AdamontScenario.rcp85
        self.gcm_rcm_couple_to_altitude_studies = {
            c: AltitudesStudies(study_class, self.altitudes,
                                gcm_rcm_couple=c, scenario=scenario) for c in gcm_rcm_couples
        }

    def test_basic_ensemble_together_fit(self):
        model_classes = [StationaryAltitudinal,
                         AltitudinalShapeConstantTimeLocationLinear,
                         AltitudinalShapeConstantTimeScaleLinear,
                         AltitudinalShapeConstantTimeLocScaleLinear
                         ][:]

        for temporal_covariate in [TimeTemporalCovariate,
                                   AnomalyTemperatureWithSplineTemporalCovariate]:
            ensemble_fit = TogetherEnsembleFit(massif_names=self.massif_names,
                                               gcm_rcm_couple_to_altitude_studies=self.gcm_rcm_couple_to_altitude_studies,
                                               models_classes=model_classes,
                                               temporal_covariate_for_fit=temporal_covariate,
                                               only_models_that_pass_goodness_of_fit_test=False)

            _ = ensemble_fit.visualizer.massif_name_to_one_fold_fit[self.massif_names[0]].best_margin_function_from_fit
        self.assertTrue(True)

    # def test_ensembe_fit_with_effect(self):
    #     model_classes = [StationaryAltitudinal][:]
    #
    #     for temporal_covariate in [TimeTemporalCovariate,
    #                                AnomalyTemperatureWithSplineTemporalCovariate]:
    #         ensemble_fit = TogetherEnsembleFit(massif_names=self.massif_names,
    #                                            gcm_rcm_couple_to_altitude_studies=self.gcm_rcm_couple_to_altitude_studies,
    #                                            models_classes=model_classes,
    #                                            temporal_covariate_for_fit=temporal_covariate,
    #                                            only_models_that_pass_goodness_of_fit_test=False,
    #                                            climate_coordinates_with_effects=[AbstractCoordinates.COORDINATE_GCM])
    #
    #         model_class_to_estimator = ensemble_fit.visualizer.massif_name_to_one_fold_fit[self.massif_names[0]].model_class_to_estimator
    #         model_class_to_expected_number_params = {
    #             StationaryAltitudinal: 5,
    #         }
    #         for model_class in model_classes:
    #             expected = model_class_to_expected_number_params[model_class]
    #             found = model_class_to_estimator[model_class].nb_params
    #             self.assertEqual(expected, found)
    #
    #         # _ = ensemble_fit.visualizer.massif_name_to_one_fold_fit[self.massif_names[0]].best_function_from_fit
    #
    #
    #     self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
