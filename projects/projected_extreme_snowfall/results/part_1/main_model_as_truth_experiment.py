import datetime
import os.path as op
import random
import time
from itertools import product

import pandas as pd

from extreme_data.utils import DATA_PATH
from projects.projected_extreme_snowfall.results.part_1.utils import load_combination_name_to_dict, \
    compute_average_nllh, load_combination_name_to_dict_v2
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates

CSV_PATH = op.join(DATA_PATH, "model_as_truth_csv")


def main_model_as_truth_experiment():
    start = time.time()

    # Parameters
    fast = None
    selection_method_names = ['aic', 'aicc', 'bic']
    climate_coordinates_with_effects_list = [None,
                                             [AbstractCoordinates.COORDINATE_GCM],
                                             [AbstractCoordinates.COORDINATE_RCM],
                                             [AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM]
                                             ]  # None means we do not create any effect
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit = set_up_and_load(fast)
    altitudes_list = [[2100], [3600]]

    gcm_rcm_couples_sampled_for_experiment = []
    for gcm in set(c[0] for c in gcm_rcm_couples):
        gcm_rcm_couple_with_this_gcm = [c for c in gcm_rcm_couples if c[0] == gcm]
        print(gcm_rcm_couple_with_this_gcm)
        gcm_rcm_couples_sampled_for_experiment.append(random.sample(gcm_rcm_couple_with_this_gcm, 1)[0])
    gcm_rcm_couples_sampled_for_experiment = [('NorESM1-M', 'REMO2015'), ('MPI-ESM-LR', 'REMO2009'),
                                              ('HadGEM2-ES', 'RCA4')]
    nb_samples_which_is_number_of_different_gcm = len(gcm_rcm_couples_sampled_for_experiment)

    print(len(gcm_rcm_couples), gcm_rcm_couples)
    print(len(gcm_rcm_couples_sampled_for_experiment), gcm_rcm_couples_sampled_for_experiment)
    # Load all the combinations
    weight_on_observation = [1, 2, 5, 10, 19, 38, 57, 76][2]
    # potential_indices = list(range(4))
    # all_combinations = [potential_indices for _ in range(3)]
    # combinations_only_for_location = [potential_indices, [0], [0]]
    # combinations_only_for_scale = [[0], potential_indices, [0]]
    # combinations_only_for_shape = [[0], [0], potential_indices]
    # combinations_list = [combinations_only_for_location, combinations_only_for_scale, combinations_only_for_shape][:1]
    # combination_name_to_dict = load_combination_name_to_dict(climate_coordinates_with_effects_list, combinations_list)

    all_couple_combinations = list(product(list(range(4)), list(range(4))))
    combinations = [(3, i, j) for i, j in all_couple_combinations][:]
    combinations = [(i,i,i) for i in range(4)][:1]
    combination_name_to_dict = load_combination_name_to_dict_v2(climate_coordinates_with_effects_list,
                                                                combinations)

    combination_names_todo = set(combination_name_to_dict.keys())
    print(combination_names_todo)

    # Load the combination already done
    altitudes_str = '_'.join([str(a[0]) for a in altitudes_list])
    csv_filename = 'fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_nb_samples_{}.csv'.format(fast,
                                                                                                         altitudes_str,
                                                                                                         len(model_classes),
                                                                                                         len(gcm_rcm_couples),
                                                                                                         nb_samples_which_is_number_of_different_gcm)
    csv_filename = 'nbloop{}_'.format(weight_on_observation) + csv_filename
    print(csv_filename)
    csv_filepath = op.join(CSV_PATH, csv_filename)
    df_csv = load_csv(csv_filepath)
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
                                       gcm_rcm_couples_sampled_for_experiment, weight_on_observation)
        combination_name_to_average_predicted_nllh = {combination_name: average}
        # Load df
        df = pd.DataFrame.from_dict(combination_name_to_average_predicted_nllh)
        df.index = selection_method_names
        df = df.transpose()
        min_column_name = 'min'
        df[min_column_name] = df.min(axis=1)
        # concat with the existing
        df_csv = load_csv(csv_filepath)
        df_csv = pd.concat([df_csv, df], axis=0)
        # sort by best scores
        df_csv.sort_values(by=min_column_name, inplace=True)
        # save intermediate results
        df_csv.to_csv(csv_filepath)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def load_csv(csv_filepath):
    if op.exists(csv_filepath):
        df_csv = pd.read_csv(csv_filepath, index_col=0)
    else:
        df_csv = pd.DataFrame()
    return df_csv


if __name__ == '__main__':
    main_model_as_truth_experiment()
