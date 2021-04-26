from typing import List

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_fit.estimator.margin_estimator.utils_functions import compute_nllh
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
                 selection_method_name='aic'
                 ):
        self.selection_method_name = selection_method_name
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

    @property
    def massif_name(self):
        assert len(self.massif_names) == 1
        return self.massif_names[0]

    def run_all_experiments(self):
        return np.mean([self.run_one_experiment(c) for c in self.gcm_rcm_couples])

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
                                                     display_only_model_that_pass_anderson_test=self.display_only_model_that_pass_gof_test,
                                                     confidence_interval_based_on_delta_method=False,
                                                     remove_physically_implausible_models=self.remove_physically_implausible_models,
                                                     climate_coordinates_with_effects=self.climate_coordinates_with_effects,
                                                     gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple_set_as_truth)

        # Get the best margin function for the selection method name
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
        best_estimator = one_fold_fit._sorted_estimators_with_method_name(self.selection_method_name)[0]
        best_margin_function_from_fit = best_estimator.margin_function_from_fit
        # Compute the average nllh for the test data
        studies = self.load_studies_for_test(gcm_rcm_couple_set_as_truth)
        dataset = studies.spatio_temporal_dataset(self.massif_name)
        df_coordinates_temp = best_estimator.load_coordinates_temp(dataset.coordinates)
        nllh = compute_nllh(df_coordinates_temp.values, dataset.observations.maxima_gev,
                            best_margin_function_from_fit)
        return nllh

    def load_studies_for_test(self, gcm_rcm_couple_set_as_truth) -> AltitudesStudies:
        """For gcm_rcm_couple_set_as_truth, load the data from 2020 to 2100"""
        return self.load_altitude_studies(gcm_rcm_couple_set_as_truth, 2006, 2100)

    def load_gcm_rcm_couple_to_studies(self, gcm_rcm_couple_set_as_truth):
        """For the gcm_rcm_couple_set_as_truth load only the data from 1959 to 2019"""
        gcm_rcm_couple_to_studies = {}
        # Load the pseudo observations
        gcm_rcm_couple_to_studies[gcm_rcm_couple_set_as_truth] = self.load_altitude_studies(gcm_rcm_couple_set_as_truth, 1959, 2005)
        # Load the rest of the projections
        for gcm_rcm_couple in set(self.gcm_rcm_couples) - {gcm_rcm_couple_set_as_truth}:
            gcm_rcm_couple_to_studies[gcm_rcm_couple] = self.load_altitude_studies(gcm_rcm_couple)
        return gcm_rcm_couple_to_studies

    def load_altitude_studies(self, gcm_rcm_couple, year_min=None, year_max=None):
        if year_min is None and year_max is None:
            kwargs = {}
        else:
            kwargs = {'year_min': year_min, 'year_max': year_max}
        return AltitudesStudies(self.study_class, self.altitudes, season=self.season,
                                   scenario=self.scenario, gcm_rcm_couple=gcm_rcm_couple, **kwargs)