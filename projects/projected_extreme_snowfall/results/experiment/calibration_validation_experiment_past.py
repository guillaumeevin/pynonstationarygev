from typing import List

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment


class CalibrationValidationExperimentPast(CalibrationValidationExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 combination=None, weight_on_observation=1, linear_effects=(False, False, False),
                 start_year_for_test_set=1990, year_max_for_studies=None):
        super().__init__(altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models, combination,
                         weight_on_observation, linear_effects, start_year_for_test_set, year_max_for_studies)

        self.start_year_for_train_set = start_year_for_test_set

    def load_studies_obs_for_test(self) -> AltitudesStudies:
        return self.load_altitude_studies(None, 1959, self.start_year_for_train_set - 1)

    def load_studies_obs_for_train(self):
        return self.load_altitude_studies(None, self.start_year_for_train_set, 2019)


class CalibrationAicExperiment(CalibrationValidationExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 combination=None, weight_on_observation=1, linear_effects=(False, False, False),
                 year_max_for_studies=None,
                 ):
        super().__init__(altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models, combination,
                         weight_on_observation, linear_effects, 2020, year_max_for_studies,
                         only_obs_score=None)

    def load_studies_obs_for_test(self) -> AltitudesStudies:
        raise NotImplementedError

    def load_studies_obs_for_train(self):
        return self.load_altitude_studies(None, 1959, 2019)
