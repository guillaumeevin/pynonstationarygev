import datetime
import time
from itertools import product

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.part_1.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.utils import load_combination_name


def compute_average_nllh(altitudes_list, param_name_to_climate_coordinates_with_effects, gcm_rcm_couples, massif_names,
                         model_classes,
                         scenario, study_class, temporal_covariate_for_fit, selection_method_names,
                         gcm_rcm_couples_sampled_for_experiment,
                         weight_on_observation, remove_physically_implausible_models, display_only_model_that_pass_gof_test):
    nllh_list = []
    for altitudes in altitudes_list:
        print('\nstart altitudes=', altitudes)
        start = time.time()

        xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                    scenario=scenario,
                                    selection_method_names=selection_method_names,
                                    model_classes=model_classes,
                                    massif_names=massif_names,
                                    fit_method=MarginFitMethod.evgam,
                                    temporal_covariate_for_fit=temporal_covariate_for_fit,
                                    remove_physically_implausible_models=remove_physically_implausible_models,
                                    display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                    gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                    weight_on_observation=weight_on_observation,
                                    )
        nllh_array = xp.run_all_experiments()
        assert len(nllh_array) == len(selection_method_names)
        nllh_list.append(nllh_array)
        end = time.time()
        duration = str(datetime.timedelta(seconds=end - start))
        print('Total duration for altitude {}m=', duration)
    return np.mean(nllh_list, axis=0)


def load_combination_name_to_dict(climate_coordinates_with_effects_list, combinations_list):
    combination_name_to_param_name_to_climate_coordinates_with_effects = {}
    for combinations in combinations_list:
        for combination in product(*combinations):
            param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                              for param_name, idx in
                                                              zip(GevParams.PARAM_NAMES, combination)}
            combination_name = load_combination_name(param_name_to_climate_coordinates_with_effects)
            combination_name_to_param_name_to_climate_coordinates_with_effects[
                combination_name] = param_name_to_climate_coordinates_with_effects
    return combination_name_to_param_name_to_climate_coordinates_with_effects

def load_combination_name_to_dict_v2(climate_coordinates_with_effects_list, combinations):
    combination_name_to_param_name_to_climate_coordinates_with_effects = {}
    for combination in combinations:
        param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                          for param_name, idx in
                                                          zip(GevParams.PARAM_NAMES, combination)}
        combination_name = load_combination_name(param_name_to_climate_coordinates_with_effects)
        combination_name_to_param_name_to_climate_coordinates_with_effects[
            combination_name] = param_name_to_climate_coordinates_with_effects
    return combination_name_to_param_name_to_climate_coordinates_with_effects