import datetime
import random
import time
from collections import OrderedDict
from itertools import product

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.utils import DATA_PATH
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.part_1.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from root_utils import VERSION
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates
import os.path as op

CSV_PATH = op.join(DATA_PATH, "model_as_truth_csv")


def main_model_as_truth_experiment():
    start = time.time()

    # Parameters
    fast = None
    selection_method_names = ['aic', 'aicc', 'bic']
    climate_coordinates_with_effects_list = [None,
                                             [AbstractCoordinates.COORDINATE_GCM],
                                             [AbstractCoordinates.COORDINATE_RCM],
                                             [AbstractCoordinates.COORDINATE_GCM,  AbstractCoordinates.COORDINATE_RCM]
                                             ]  # None means we do not create any effect
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit = set_up_and_load(fast)

    nb_samples = min(3, len(gcm_rcm_couples))
    gcm_rcm_couples_sampled_for_experiment = random.sample(gcm_rcm_couples, k=nb_samples)
    print(gcm_rcm_couples_sampled_for_experiment)

    # Load all the combinations
    combination_name_to_dict = load_combination_name_to_dict(climate_coordinates_with_effects_list)
    combination_names_todo = set(combination_name_to_dict.keys())
    # Load the combination already done
    altitudes_str = '_'.join([str(a[0]) for a in altitudes_list])
    csv_filename = 'fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_nb_samples_{}.csv'.format(fast, altitudes_str,
                                                                                           len(model_classes),
                                                                                           len(gcm_rcm_couples),
                                                                                                         nb_samples)
    print(csv_filename)
    csv_filepath = op.join(CSV_PATH, csv_filename)
    if op.exists(csv_filepath):
        df_csv = pd.read_csv(csv_filepath, index_col=0)
    else:
        df_csv = pd.DataFrame()
    combination_names_already_done = set(df_csv.index)
    combination_names_for_loop = combination_names_todo - combination_names_already_done
    print('Nb of combination done: {}/{}'.format(len(combination_names_todo) - len(combination_names_for_loop),
                                                 len(combination_names_todo)))
    for i, combination_name in enumerate(combination_names_for_loop, 1):
        print('\n\n\nCombination running:', combination_name, ' {}/{}'.format(i, len(combination_names_for_loop)))
        param_name_to_climate_coordinates_with_effects = combination_name_to_dict[combination_name]
        average = compute_average_nllh(altitudes_list, param_name_to_climate_coordinates_with_effects, gcm_rcm_couples,
                                       massif_names,
                                       model_classes, scenario, study_class, temporal_covariate_for_fit,
                                       selection_method_names,
                                       gcm_rcm_couples_sampled_for_experiment)
        combination_name_to_average_predicted_nllh = {combination_name: average}
        # Load df
        df = pd.DataFrame.from_dict(combination_name_to_average_predicted_nllh)
        df.index = selection_method_names
        df = df.transpose()
        min_column_name = 'min'
        df[min_column_name] = df.min(axis=1)
        # concat with the existing
        df_csv = pd.concat([df_csv, df], axis=0)
        # sort by best scores
        df_csv.sort_values(by=min_column_name, inplace=True)
        # save intermediate results
        df_csv.to_csv(csv_filepath)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def load_combination_name_to_dict(climate_coordinates_with_effects_list):
    potential_indices = list(range(4))
    combination_name_to_param_name_to_climate_coordinates_with_effects = {}
    all_combinations = [potential_indices for _ in range(3)]
    combinations_only_for_location = [potential_indices, [0], [0]]
    combinations_only_for_scale = [[0], potential_indices, [0]]
    combinations_only_for_shape = [[0], [0], potential_indices]
    combinations_list = [combinations_only_for_location, combinations_only_for_scale, combinations_only_for_shape]
    for combinations in combinations_list:
        for combination in product(*combinations):
            param_name_to_climate_coordinates_with_effects = {param_name: climate_coordinates_with_effects_list[idx]
                                                              for param_name, idx in
                                                              zip(GevParams.PARAM_NAMES, combination)}
            combination_name = load_combination_name(param_name_to_climate_coordinates_with_effects)
            combination_name_to_param_name_to_climate_coordinates_with_effects[
                combination_name] = param_name_to_climate_coordinates_with_effects
    return combination_name_to_param_name_to_climate_coordinates_with_effects


def load_combination_name(param_name_to_climate_coordinates_with_effects):
    param_name_to_effect_name = {p: '+'.join([e.replace('coord_', '') for e in l])
                                 for p, l in param_name_to_climate_coordinates_with_effects.items() if
                                 l is not None}
    combination_name = ' '.join(
        [param_name + '_' + param_name_to_effect_name[param_name] for param_name in GevParams.PARAM_NAMES
         if param_name in param_name_to_effect_name])
    if combination_name == '':
        combination_name = 'no effect'
    return combination_name


def compute_average_nllh(altitudes_list, param_name_to_climate_coordinates_with_effects, gcm_rcm_couples, massif_names,
                         model_classes,
                         scenario, study_class, temporal_covariate_for_fit, selection_method_names, gcm_rcm_couples_sampled_for_experiment):
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
                                    remove_physically_implausible_models=True,
                                    display_only_model_that_pass_gof_test=True,
                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                    gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment
                                    )
        nllh_array = xp.run_all_experiments()
        assert len(nllh_array) == len(selection_method_names)
        nllh_list.append(nllh_array)
        end = time.time()
        duration = str(datetime.timedelta(seconds=end - start))
        print('Total duration for altitude {}m=', duration)
    return np.mean(nllh_list, axis=0)


if __name__ == '__main__':
    main_model_as_truth_experiment()
