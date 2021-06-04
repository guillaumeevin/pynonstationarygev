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
from root_utils import get_display_name_from_object_type


class AbstractExperiment(object):

    def __init__(self, altitudes, gcm_rcm_couples, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 selection_method_names: List[str],
                massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 remove_physically_implausible_models=False,
                 param_name_to_climate_coordinates_with_effects=None,
                 ):
        self.selection_method_names = selection_method_names
        self.fit_method = fit_method
        self.massif_names = massif_names
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.param_name_to_climate_coordinates_with_effects = param_name_to_climate_coordinates_with_effects
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

    def run_one_experiment(self, **kwargs):
        start = time.time()
        try:
            nllh_list = self._run_one_experiment(kwargs)
        except (NllhIsInfException, SafeRunException) as e:
            print(e.__repr__())
            nllh_list = [np.nan for _ in self.selection_method_names]

        duration = str(datetime.timedelta(seconds=time.time() - start))
        print('Total duration for one experiment', duration)

        return np.array(nllh_list)

    def _run_one_experiment(self, kwargs):
        # Load gcm_rcm_couple_to_studies
        gcm_rcm_couple_to_studies = self.load_gcm_rcm_couple_to_studies(**kwargs)
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
                                                     param_name_to_climate_coordinates_with_effects=self.param_name_to_climate_coordinates_with_effects,
                                                     **kwargs)
        # Get the best margin function for the selection method name
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
        best_estimator_list = [one_fold_fit._sorted_estimators_with_method_name(selection_method_name)[0]
                               for selection_method_name in self.selection_method_names]
        # Compute the average nllh for the test data
        studies_for_test = self.load_studies_for_test(**kwargs)
        dataset_test = self.load_spatio_temporal_dataset(studies_for_test, **kwargs)
        nllh_list = []
        for best_estimator in best_estimator_list:
            df_coordinates_temp_for_test = best_estimator.load_coordinates_temp(dataset_test.coordinates)
            nllh_for_test = compute_nllh(df_coordinates_temp_for_test.values, dataset_test.observations.maxima_gev,
                                best_estimator.margin_function_from_fit)
            nllh_list.append(nllh_for_test)
        return nllh_list

    def load_spatio_temporal_dataset(self, studies, **kwargs):
        raise NotImplementedError

    def load_studies_for_test(self, **kwargs) -> AltitudesStudies:
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies(self, **kwargs):
        raise NotImplementedError

    def load_altitude_studies(self, gcm_rcm_couple, year_min=None, year_max=None):
        raise NotImplementedError