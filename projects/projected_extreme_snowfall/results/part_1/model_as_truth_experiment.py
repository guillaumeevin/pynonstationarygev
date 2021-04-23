from typing import List

from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble


class ModelAsTruthExperiment(object):

    def __init__(self, altitudes, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 remove_physically_implausible_models=False,
                 climate_coordinates_with_effects=None,
                 ):
        self.fit_method = fit_method
        self.massif_names = massif_names
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.climate_coordinates_with_effects = climate_coordinates_with_effects
        self.model_classes = model_classes
        self.scenario = scenario
        self.season = season
        self.study_class = study_class
        self.gcm_rcm_couples = gcm_rcm_couples
        self.altitudes = altitudes

    def run_all_experiments(self):
        pass

    def run_one_experiment(self, gcm_rcm_couple_set_as_truth):
        # Load gcm_rcm_couple_to_studies
        gcm_rcm_couple_to_studies = self.load_gcm_rcm_couple_to_studies(gcm_rcm_couple_set_as_truth)
        # Load ensemble fit
        visualizer = VisualizerNonStationaryEnsemble(gcm_rcm_couple_to_studies=gcm_rcm_couple_to_studies,
                                                   model_classes=self.model_classes,
                                                       show=False,
                                                       massif_names=self.massif_names,
                                                  fit_method=self.fit_method,
                                                   temporal_covariate_for_fit=self.temporal_covariate_for_fit,
                                                  display_only_model_that_pass_gof_test=self.display_only_model_that_pass_gof_test,
                                                  confidence_interval_based_on_delta_method=False,
                                                  remove_physically_implausible_models=self.remove_physically_implausible_models,
                                                  climate_coordinates_with_effects=self.climate_coordinates_with_effects)
        # Compute the average nllh for the test data
        studies = self.load_gcm_rcm_couple_to_studies(gcm_rcm_couple_set_as_truth)

    def load_studies_for_test(self, gcm_rcm_couple_set_as_truth):
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        pass

    def load_gcm_rcm_couple_to_studies(self, gcm_rcm_couple_set_as_truth):
        """For the gcm_rcm_couple_set_as_truth load only the data from 1959 to 2019"""
        pass
        # kwargs_study = {'year_min': year_min, 'year_max': year_max}
        #
        # studies = AltitudesStudies(study_class, altitudes, season=season,
        #                            scenario=scenario, gcm_rcm_couple=gcm_rcm_couple,
        #                            **kwargs_study)
        # gcm_rcm_couple_to_studies[gcm_rcm_couple] = studies
        # # Potentially add the observations
        #
        #
        # if self.safran_study_class is not None:
        #     studies = AltitudesStudies(self.safran_study_class, altitudes, season=season)
        #     gcm_rcm_couple_to_studies[(None, None)] = studies