from typing import List

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projected_extremes.section_results.validation_experiment.abstract_experiment import AbstractExperiment


class ModelAsTruthExperiment(AbstractExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 combination=None,
                 gcm_rcm_couples_sampled_for_experiment=None, weight_on_observation=1,
                 linear_effects=(False, False, False),
                 year_max_for_gcm=2100,
                 year_max_for_pseudo_obs=2019,
                 ):
        super().__init__(altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models,
                         combination, weight_on_observation, linear_effects)
        self.year_max_for_pseudo_obs = year_max_for_pseudo_obs
        self.year_max = year_max_for_gcm
        self.gcm_rcm_couples_sampled_for_experiment = gcm_rcm_couples_sampled_for_experiment

    @property
    def specific_folder(self):
        return "{} {}".format(self.altitude, self.variable_name)

    @property
    def excel_filename(self):
        return super().excel_filename + '_{}_{}'.format(self.year_max, self.year_max_for_pseudo_obs)

    def load_studies_obs_for_test(self, gcm_rcm_couple_as_pseudo_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, self.year_max_for_pseudo_obs + 1,
                                          self.year_max)

    def load_studies_obs_for_train(self, gcm_rcm_couple_as_pseudo_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, 1959, self.year_max_for_pseudo_obs)

    def gcm_rcm_couples_for_ensemble_members(self, gcm_rcm_couple_as_pseudo_truth):
        return set(self.gcm_rcm_couples) - {gcm_rcm_couple_as_pseudo_truth}

    def load_gcm_rcm_couple_to_studies(self, gcm_rcm_couple_as_pseudo_truth):
        """For the gcm_rcm_couple_set_as_truth load only the data from 1959 to 2019"""
        gcm_rcm_couple_to_studies = {}
        # Load the pseudo observations
        if self.add_observations_to_gcm_rcm_couple_to_studies:
            pseudo_truth_studies = self.load_studies_obs_for_train(gcm_rcm_couple_as_pseudo_truth)
            assert pseudo_truth_studies.study.year_min == 1959, pseudo_truth_studies.study.year_min
            assert pseudo_truth_studies.study.year_max <= 2019, pseudo_truth_studies.study.year_max
            gcm_rcm_couple_to_studies[gcm_rcm_couple_as_pseudo_truth] = pseudo_truth_studies
        # Load the rest of the projections
        for gcm_rcm_couple in self.gcm_rcm_couples_for_ensemble_members(gcm_rcm_couple_as_pseudo_truth):
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple,
                                                                                   year_max=self.year_max)
        return gcm_rcm_couple_to_studies

    def load_gcm_rcm_couple_to_studies_for_ensemble_members(self, gcm_rcm_couple_as_pseudo_truth):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in self.gcm_rcm_couples_for_ensemble_members(gcm_rcm_couple_as_pseudo_truth):
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple,
                                                                                   year_max=self.year_max)
        return gcm_rcm_couple_to_studies

    def load_gcm_rcm_couple_to_studies_for_train_period_and_ensemble_members(self, gcm_rcm_couple_as_pseudo_truth):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in self.gcm_rcm_couples_for_ensemble_members(gcm_rcm_couple_as_pseudo_truth):
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple,
                                                                                   year_max=2019)
        return gcm_rcm_couple_to_studies

    def load_gcm_rcm_couple_to_studies_for_test_period_and_ensemble_members(self, gcm_rcm_couple_as_pseudo_truth):
        gcm_rcm_couple_to_studies = {}
        for gcm_rcm_couple in self.gcm_rcm_couples_for_ensemble_members(gcm_rcm_couple_as_pseudo_truth):
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple,
                                                                                   year_min=2020,
                                                                                   year_max=2100)
        return gcm_rcm_couple_to_studies
