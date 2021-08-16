import datetime
import random
import time
from typing import List

import numpy as np
from rpy2.rinterface import RRuntimeError

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh, NllhIsInfException
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import SafeRunException
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from projects.projected_extreme_snowfall.results.experiment.abstract_experiment import AbstractExperiment
from root_utils import get_display_name_from_object_type


class CalibrationValidationExperiment(AbstractExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 start_year_for_test_set=1990,
                 year_max_for_studies=None):
        super().__init__(altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models,
                         param_name_to_climate_coordinates_with_effects)
        self.year_max_for_studies = year_max_for_studies
        self.start_year_for_test_set = start_year_for_test_set

    @property
    def excel_filename(self):
        return super().excel_filename + '_{}_{}'.format(self.start_year_for_test_set-1, self.year_max_for_studies)

    def load_studies_obs_for_test(self) -> AltitudesStudies:
        return self.load_altitude_studies(None, self.start_year_for_test_set, 2019)

    def load_studies_obs_for_train(self):
        return self.load_altitude_studies(None, 1959, self.start_year_for_test_set-1)

    def load_gcm_rcm_couple_to_studies(self):
        gcm_rcm_couple_to_studies = {}
        # Load the pseudo observations
        gcm_rcm_couple_to_studies[(None, None)] = self.load_studies_obs_for_train()
        # Load the rest of the projections
        for gcm_rcm_couple in self.gcm_rcm_couples:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple, None, year_max=self.year_max_for_studies)
        return gcm_rcm_couple_to_studies

    def load_gcm_rcm_couple_to_studies_for_train_period_and_ensemble_members(self, **kwargs):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in self.gcm_rcm_couples:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple, None, year_max=self.start_year_for_test_set-1)
        return gcm_rcm_couple_to_studies

    def load_gcm_rcm_couple_to_studies_for_test_period_and_ensemble_members(self, **kwargs):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in self.gcm_rcm_couples:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple, year_min=self.start_year_for_test_set, year_max=2019)
        return gcm_rcm_couple_to_studies




