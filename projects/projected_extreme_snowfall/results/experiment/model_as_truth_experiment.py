from typing import List

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.experiment.abstract_experiment import AbstractExperiment


class ModelAsTruthExperiment(AbstractExperiment):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel], selection_method_names: List[str],
                 massif_names=None, fit_method=MarginFitMethod.extremes_fevd_mle, temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False, remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 gcm_rcm_couples_sampled_for_experiment=None, weight_on_observation=1,
                 year_max_for_gcm=2100,
                 year_max_for_pseudo_obs=2019,
                 ):
        super().__init__(altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario, model_classes,
                         selection_method_names, massif_names, fit_method, temporal_covariate_for_fit,
                         display_only_model_that_pass_gof_test, remove_physically_implausible_models,
                         param_name_to_climate_coordinates_with_effects)
        self.year_max_for_pseudo_obs = year_max_for_pseudo_obs
        self.year_max = year_max_for_gcm
        self.gcm_rcm_couples_sampled_for_experiment = gcm_rcm_couples_sampled_for_experiment
        self.weight_on_observation = weight_on_observation

    @property
    def kwargs_for_visualizer(self):
        return {'weight_on_observation': self.weight_on_observation}

    # def plot_time_series(self):
    #     # plot time series
    #     gcm_rcm_couple_to_studies_plot = xp.load_gcm_rcm_couple_to_studies(
    #         gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)
    #     gcm_rcm_couple_to_study_plot = {c: studies.study for c, studies in
    #                                     gcm_rcm_couple_to_studies_plot.items()}
    #     gcm_rcm_couple_to_other_study_plot = {c: s for c, s in gcm_rcm_couple_to_study.items() if
    #                                           c != gcm_rcm_couple}
    #     plot_time_series(massif_name, gcm_rcm_couple_to_study_plot[gcm_rcm_couple],
    #                      gcm_rcm_couple_to_other_study_plot, show)



    @property
    def excel_filename(self):
        return super().excel_filename + '_{}_{}_w{}'.format(self.year_max_for_pseudo_obs, self.year_max, self.weight_on_observation)

    def load_studies_obs_for_test(self, gcm_rcm_couple_as_pseudo_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, self.year_max_for_pseudo_obs+1, self.year_max)

    def load_studies_obs_for_train(self, gcm_rcm_couple_as_pseudo_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_as_pseudo_truth, 1959, self.year_max_for_pseudo_obs)

    def load_gcm_rcm_couple_to_studies(self, gcm_rcm_couple_as_pseudo_truth):
        """For the gcm_rcm_couple_set_as_truth load only the data from 1959 to 2019"""
        gcm_rcm_couple_to_studies = {}
        # Load the pseudo observations
        pseudo_truth_studies = self.load_studies_obs_for_train(gcm_rcm_couple_as_pseudo_truth)
        assert pseudo_truth_studies.study.year_min == 1959, pseudo_truth_studies.study.year_min
        assert pseudo_truth_studies.study.year_max <= 2019, pseudo_truth_studies.study.year_max
        gcm_rcm_couple_to_studies[gcm_rcm_couple_as_pseudo_truth] = pseudo_truth_studies

        # Load the rest of the projections
        for gcm_rcm_couple in set(self.gcm_rcm_couples) - {gcm_rcm_couple_as_pseudo_truth}:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple,
                                                                                   year_max=self.year_max)
        return gcm_rcm_couple_to_studies
