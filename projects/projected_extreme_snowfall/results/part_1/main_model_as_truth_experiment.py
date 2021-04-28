import datetime
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
    fast = True
    selection_method_names = ['aic', 'aicc', 'bic']
    climate_coordinates_with_effects_list = [[AbstractCoordinates.COORDINATE_GCM, AbstractCoordinates.COORDINATE_RCM],
                                             [AbstractCoordinates.COORDINATE_GCM],
                                             [AbstractCoordinates.COORDINATE_RCM],
                                             None
                                             ]  # None means we do not create any effect
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit = set_up_and_load(fast)

    # Load all the combinations
    combination_name_to_dict = load_combination_name_to_dict(climate_coordinates_with_effects_list)
    combination_names_todo = set(combination_name_to_dict.keys())
    # Load the combination already done
    csv_filepath = op.join(CSV_PATH, 'fast_{}.csv'.format(fast))
    if op.exists(csv_filepath):
        df_csv = pd.read_csv(csv_filepath)
    else:
        df_csv = pd.DataFrame()
    combination_names_already_done = set(df_csv.index)
    print('Nb of combination done: {}/{}'.format(len(combination_names_already_done), len(combination_names_todo)))
    combination_names_for_loop = combination_names_todo - combination_names_already_done
    for i, combination_name in enumerate(combination_names_for_loop, 1):
        print('Combination running:', combination_name, ' {}/{}'.format(i, len(combination_names_for_loop)))
        param_name_to_climate_coordinates_with_effects = combination_name_to_dict[combination_name]
        average = compute_average_nllh(altitudes_list, param_name_to_climate_coordinates_with_effects, gcm_rcm_couples,
                                       massif_names,
                                       model_classes, scenario, study_class, temporal_covariate_for_fit,
                                       selection_method_names)
        combination_name_to_average_predicted_nllh = {combination_name: average}
        # Load df
        df = pd.DataFrame.from_dict(combination_name_to_average_predicted_nllh)
        df.index = selection_method_names
        df = df.transpose()
        df['max'] = df.max(axis=1)
        # concat with the existing
        df_csv = pd.concat([df_csv, df], axis=0)
        # sort by best scores
        df_csv.sort_values(by='max', inplace=True, ascending=False)
        # save intermediate results
        df_csv.to_csv(csv_filepath)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def load_combination_name_to_dict(climate_coordinates_with_effects_list):
    potential_indices = list(range(4))
    combination_name_to_param_name_to_climate_coordinates_with_effects = {}
    for combination in product(*[potential_indices for _ in range(3)]):
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
                         scenario, study_class, temporal_covariate_for_fit, selection_method_names):
    nllh_list = []
    for altitudes in altitudes_list:
        print('altitudes=', altitudes)
        xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                    scenario=scenario,
                                    selection_method_names=selection_method_names,
                                    model_classes=model_classes,
                                    massif_names=massif_names,
                                    fit_method=MarginFitMethod.evgam,
                                    temporal_covariate_for_fit=temporal_covariate_for_fit,
                                    remove_physically_implausible_models=True,
                                    display_only_model_that_pass_gof_test=True,
                                    param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects
                                    )
        nllh_array = xp.run_all_experiments()
        assert len(nllh_array) == len(selection_method_names)
        nllh_list.append(nllh_array)
    return np.mean(nllh_list, axis=0)


if __name__ == '__main__':
    main_model_as_truth_experiment()
