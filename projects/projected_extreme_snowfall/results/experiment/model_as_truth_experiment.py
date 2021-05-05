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


class ModelAsTruthExperiment(AbstractExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 gcm_rcm_couples_sampled_for_experiment=None, weight_on_observation=1
                 ):
        super().__init__(altitudes, gcm_rcm_couples, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models,
                         param_name_to_climate_coordinates_with_effects)
        self.gcm_rcm_couples_sampled_for_experiment = gcm_rcm_couples_sampled_for_experiment
        self.weight_on_observation = weight_on_observation

    @property
    def massif_name(self):
        assert len(self.massif_names) == 1
        return self.massif_names[0]

    def run_all_experiments(self):
        return np.nanmean([self.run_one_experiment(c) for c in self.gcm_rcm_couples_sampled_for_experiment], axis=0)

    def load_spatio_temporal_dataset(self, studies, **kwargs):

        return studies.spatio_temporal_dataset(self.massif_name, **kwargs)

    def load_studies_for_test(self, gcm_rcm_couple_as_pseudo_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, 2020, 2100)

    def load_gcm_rcm_couple_to_studies(self, gcm_rcm_couple_as_pseudo_truth):
        """For the gcm_rcm_couple_set_as_truth load only the data from 1959 to 2019"""
        gcm_rcm_couple_to_studies = {}
        # Load the pseudo observations
        pseudo_truth_studies = self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, 1959, 2019)
        assert pseudo_truth_studies.study.year_min == 1959
        assert pseudo_truth_studies.study.year_max == 2019
        gcm_rcm_couple_to_studies[gcm_rcm_couple_as_pseudo_truth] = pseudo_truth_studies

        # Load the rest of the projections
        for gcm_rcm_couple in set(self.gcm_rcm_couples) - {gcm_rcm_couple_as_pseudo_truth}:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple)
        return gcm_rcm_couple_to_studies

    def load_altitude_studies(self, gcm_rcm_couple, year_min=None, year_max=None):
        if year_min is None and year_max is None:
            kwargs = {}
        else:
            kwargs = {'year_min': year_min, 'year_max': year_max}
        return AltitudesStudies(self.study_class, self.altitudes, season=self.season,
                                scenario=self.scenario, gcm_rcm_couple=gcm_rcm_couple, **kwargs)
