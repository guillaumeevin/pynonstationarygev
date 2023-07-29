import datetime
import itertools
import os
import os.path as op
import time
from typing import List

import numpy as np
import numpy.testing as npt

from extreme_data.meteo_france_data.scm_models_data.altitudes_studies import AltitudesStudies
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_precipf import SafranPrecipitation2019
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2019, \
    SafranSnowfall3Days2022, SafranSnowfall5Days2022
from extreme_data.utils import RESULTS_PATH
from extreme_fit.estimator.margin_estimator.utils_functions import NllhIsInfException, compute_nllh
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.utils import SafeRunException
from extreme_trend.ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects, load_combination_name
from projected_extremes.section_results.utils.csv_utils import update_csv, is_already_done
from root_utils import get_display_name_from_object_type


class AbstractExperiment(object):

    def __init__(self, altitudes, gcm_rcm_couples, safran_study_class, study_class, season, scenario,
                 model_classes: List[AbstractTemporalLinearMarginModel],
                 selection_method_names: List[str],
                 massif_names=None,
                 fit_method=MarginFitMethod.extremes_fevd_mle,
                 temporal_covariate_for_fit=None,
                 display_only_model_that_pass_gof_test=False,
                 remove_physically_implausible_models=False,
                 combination=None,
                 weight_on_observation=1,
                 linear_effects=(False, False, False),
                 only_obs_score=True,
                 ):
        self.only_obs_score = only_obs_score
        self.weight_on_observation = weight_on_observation
        self.linear_effects = linear_effects
        self.selection_method_names = selection_method_names
        self.fit_method = fit_method
        self.massif_names = massif_names
        self.temporal_covariate_for_fit = temporal_covariate_for_fit
        self.display_only_model_that_pass_gof_test = display_only_model_that_pass_gof_test
        self.remove_physically_implausible_models = remove_physically_implausible_models
        self.combination = combination
        self.model_classes = model_classes
        self.scenario = scenario
        self.season = season
        self.study_class = study_class
        self.safran_study_class = safran_study_class
        self.gcm_rcm_couples = gcm_rcm_couples
        self.altitudes = altitudes

    @property
    def param_name_to_climate_coordinates_with_effects(self):
        if -1 in self.combination:
            return None
        else:
            return load_param_name_to_climate_coordinates_with_effects(self.combination)

    @property
    def add_observations_to_gcm_rcm_couple_to_studies(self):
        return -1 not in self.combination

    def load_studies_obs_for_train(self, **kwargs) -> AltitudesStudies:
        raise NotImplementedError

    def load_studies_obs_for_test(self, **kwargs) -> AltitudesStudies:
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies(self, **kwargs):
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies_for_ensemble_members(self, **kwargs):
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies_for_train_period_and_ensemble_members(self, **kwargs):
        raise NotImplementedError

    def load_gcm_rcm_couple_to_studies_for_test_period_and_ensemble_members(self, **kwargs):
        raise NotImplementedError

    def run_one_experiment(self, **kwargs):
        if len(kwargs) == 0:
            if len(self.gcm_rcm_couples) > 1:
                gcm_rcm_couple = ("", "")
            else:
                gcm_rcm_couple = self.gcm_rcm_couples[0]
        else:
            gcm_rcm_couple = kwargs['gcm_rcm_couple_as_pseudo_truth']
        is_already_done_list = [
            is_already_done(self.excel_filepath, self.get_row_name(p), self.experiment_name, gcm_rcm_couple) for p
            in self.prefixs]
        is_already_done_all = all(is_already_done_list)
        if not is_already_done_all:
            start = time.time()
            try:
                nllh_lists = self._run_one_experiment(kwargs)
            except (NllhIsInfException, SafeRunException, KeyError) as e:
                print(e.__repr__())
                nllh_lists = [[np.nan] for _ in self.prefixs]
            duration = str(datetime.timedelta(seconds=time.time() - start))
            for nllh_list, prefix in zip(nllh_lists, self.prefixs):
                row_name = self.get_row_name(prefix)
                update_csv(self.excel_filepath, row_name, self.experiment_name, gcm_rcm_couple, np.array(nllh_list))

    @property
    def kwargs_for_visualizer(self):
        return {'weight_on_observation': self.weight_on_observation,
                'linear_effects': self.linear_effects}

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
                                                     **kwargs,
                                                     **self.kwargs_for_visualizer)
        # Get the best margin function for the selection method name
        one_fold_fit = visualizer.massif_name_to_one_fold_fit[self.massif_name]
        assert len(one_fold_fit.fitted_estimators) == 1, 'for the model as truth they should not be any combinations'
        assert len(self.selection_method_names) == 1
        best_estimator = one_fold_fit._sorted_estimators_with_method_name("aic")[0]
        # Compute the log score for the observations
        if self.only_obs_score is True:
            gumbel_standardization = False
            studies_for_train = self.load_studies_obs_for_train(**kwargs)
            studies_for_test = self.load_studies_obs_for_test(**kwargs)
            train_nllh_list = self.compute_nllh_list(best_estimator, kwargs, studies_for_train, gumbel_standardization)
            test_nllh_list = self.compute_nllh_list(best_estimator, kwargs, studies_for_test, gumbel_standardization)
            return [train_nllh_list, test_nllh_list]
        elif self.only_obs_score is None:
            return [best_estimator.aic, best_estimator.aic]
        else:
            gumbel_standardization = False
            if gumbel_standardization:
                # Check the nllh
                _, _, total_nllh_list, _ = self.compute_log_score(best_estimator, False, kwargs)
                assert len(total_nllh_list) == len(best_estimator.coordinates_for_nllh)
                npt.assert_almost_equal(sum(total_nllh_list), best_estimator.nllh, decimal=0)
            ensemble_members_nllh_list, test_nllh_list, total_nllh_list, train_nllh_list = self.compute_log_score(
                best_estimator,
                gumbel_standardization, kwargs)
            return [train_nllh_list, test_nllh_list, ensemble_members_nllh_list, total_nllh_list]

    def compute_log_score(self, best_estimator, gumbel_standardization, kwargs):
        studies_for_train = self.load_studies_obs_for_train(**kwargs)
        studies_for_test = self.load_studies_obs_for_test(**kwargs)
        studies_list_for_ensemble_members = self.load_gcm_rcm_couple_to_studies_for_ensemble_members(
            **kwargs).values()

        train_nllh_list = self.compute_nllh_list(best_estimator, kwargs, studies_for_train, gumbel_standardization)
        test_nllh_list = self.compute_nllh_list(best_estimator, kwargs, studies_for_test, gumbel_standardization)
        ensemble_members_nllh_list = self.compute_nllh_list_from_studies_list(best_estimator, kwargs,
                                                                              studies_list_for_ensemble_members,
                                                                              gumbel_standardization)
        train_nllh_list = list(
            itertools.chain.from_iterable([train_nllh_list for _ in range(self.weight_on_observation)]))
        total_nllh_list = train_nllh_list + ensemble_members_nllh_list
        return ensemble_members_nllh_list, test_nllh_list, total_nllh_list, train_nllh_list

    def compute_nllh_list_from_studies_list(self, best_estimator, kwargs, studies_list, gumbel_standardization):
        gcm_rcm_studies_nllh_list = []
        for studies in studies_list:
            nllh_list = self.compute_nllh_list(best_estimator, kwargs, studies, gumbel_standardization)
            gcm_rcm_studies_nllh_list.extend(nllh_list)
        return gcm_rcm_studies_nllh_list

    def compute_nllh_list(self, best_estimator, kwargs, studies, gumbel_standardization):
        dataset_test = self.load_spatio_temporal_dataset(studies, **kwargs)
        df_coordinates_temp_for_test = best_estimator.load_coordinates_temp(dataset_test.coordinates,
                                                                            for_fit=False)
        maxima_values = dataset_test.observations.maxima_gev
        coordinate_values = df_coordinates_temp_for_test.values
        nllh = compute_nllh(coordinate_values, maxima_values, best_estimator.margin_function_from_fit,
                                                                   True, True, gumbel_standardization)
        return [nllh / len(coordinate_values) for _ in coordinate_values]

    def load_altitude_studies(self, gcm_rcm_couple=None, year_min=None, year_max=None):
        kwargs = {}
        if year_min is not None:
            kwargs['year_min'] = year_min
        if year_max is not None:
            kwargs['year_max'] = year_max
        if gcm_rcm_couple is None:
            return AltitudesStudies(self.safran_study_class, self.altitudes, season=self.season, **kwargs)
        else:
            return AltitudesStudies(self.study_class, self.altitudes, season=self.season,
                                    scenario=self.scenario, gcm_rcm_couple=gcm_rcm_couple, **kwargs)

    # Utils

    def load_spatio_temporal_dataset(self, studies, **kwargs):
        return studies.spatio_temporal_dataset(self.massif_name, **kwargs)

    @property
    def excel_filename(self):
        study_name = get_display_name_from_object_type(self.study_class)
        altitude = str(self.altitude)
        nb_couples = len(self.gcm_rcm_couples)
        goodness_of_fit = self.display_only_model_that_pass_gof_test
        model_name = get_display_name_from_object_type(self.model_class)
        return "{}_{}m_{}couples_test{}_{}_w{}_{}".format(study_name, altitude, nb_couples, goodness_of_fit, model_name,
                                                          self.weight_on_observation,
                                                          self.linear_effects)

    @property
    def excel_filepath(self):
        path = op.join(RESULTS_PATH, "abstract_experiments",
                       get_display_name_from_object_type(self),
                       self.specific_folder)
        if not op.exists(path):
            os.makedirs(path, exist_ok=True)
        excel_filepath = op.join(path, self.excel_filename + '.xlsx')
        excel_filepath = excel_filepath.replace(' ', '')
        return excel_filepath

    @property
    def specific_folder(self):
        raise NotImplementedError

    @property
    def variable_name(self):
        if self.safran_study_class is CrocusSnowLoad2019:
            return "snow load"
        elif self.safran_study_class is SafranSnowfall2019:
            return "snowfall"
        elif self.safran_study_class is SafranPrecipitation2019:
            return "precipitation"
        elif self.safran_study_class is SafranSnowfall3Days2022:
            return "snowfall3days2022"
        elif self.safran_study_class is SafranSnowfall5Days2022:
            return "snowfall5days2022"
        else:
            raise NotImplementedError

    @property
    def massif_name(self):
        assert len(self.massif_names) == 1
        return self.massif_names[0]

    @property
    def model_class(self):
        assert len(self.model_classes) == 1
        return self.model_classes[0]

    @property
    def altitude(self):
        assert len(self.altitudes) == 1
        return self.altitudes[0]

    @property
    def experiment_name(self):
        return str(self.altitude) + self.massif_name

    @property
    def combination_name(self):
        if self.add_observations_to_gcm_rcm_couple_to_studies:
            return "with obs and " + load_combination_name(self.param_name_to_climate_coordinates_with_effects)
        else:
            return "without obs"

    def get_row_name(self, prefix):
        return "{}_{}".format(prefix, self.combination_name)

    @property
    def prefixs(self):
        prefixs = ['CalibrationObs', 'ValidationObs', 'CalibrationEnsembleMembers', 'CalibrationAll']
        if self.only_obs_score is True:
            return prefixs[:2]
        elif self.only_obs_score is None:
            return prefixs[:2]
        else:
            return prefixs
