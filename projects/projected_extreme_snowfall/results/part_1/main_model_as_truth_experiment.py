import datetime
import os.path as op
import random
import time
from itertools import product

import numpy as np
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.utils import DATA_PATH
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.part_1.compute_pairwise_distance import \
    ordered_gcm_rcm_couples_in_terms_of_bias_similar_to_bias_of_obs
from projects.projected_extreme_snowfall.results.part_1.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_1.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.part_1.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_1.v1.utils_v1 import compute_average_nllh, \
    load_combination_name_to_dict_v2
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from projects.projected_extreme_snowfall.results.utils import climate_coordinates_with_effects_list, \
    load_combination_name_for_tuple, load_param_name_to_climate_coordinates_with_effects


def main_model_as_truth_experiment():
    start = time.time()

    fast = False
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class = set_up_and_load \
        (fast)

    nb_gcm_rcm_couples_as_truth_that_are_the_closest = 1

    # Load the csv filepath
    altitudes_str = '_'.join([str(a[0]) for a in altitudes_list])
    csv_filename = 'fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_nb_samples_{}.csv'.format(fast,
                                                                                                         altitudes_str,
                                                                                                         len(model_classes),
                                                                                                         len(gcm_rcm_couples),
                                                                                                         nb_gcm_rcm_couples_as_truth_that_are_the_closest)
    csv_filepath = op.join(CSV_PATH, csv_filename)

    combinations = [(i, i, i) for i in [3]]
    potential_indices = list(range(4))
    couples = list(product(potential_indices, potential_indices))
    combinations = [tuple([3] + list(c)) for c in couples]

    for altitudes in altitudes_list:
        altitude = altitudes[0]

        gcm_rcm_running = ordered_gcm_rcm_couples_in_terms_of_bias_similar_to_bias_of_obs(
            nb_gcm_rcm_couples_as_truth_that_are_the_closest, altitude,
            gcm_rcm_couples, massif_names, scenario,
            1959, 2019, study_class, safran_study_class)

        for gcm_rcm_couple in gcm_rcm_running:
            print("Running:", altitude, gcm_rcm_running)
            for i, combination in enumerate(combinations, 1):
                print('Combination {}/{}'.format(i, len(combinations)))

                combination_name = load_combination_name_for_tuple(combination)
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                    combination)

                if is_already_done(csv_filepath, combination_name, altitude, gcm_rcm_couple):
                    continue

                xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                            scenario=scenario,
                                            selection_method_names=['aic'],
                                            model_classes=model_classes,
                                            massif_names=massif_names,
                                            fit_method=MarginFitMethod.evgam,
                                            temporal_covariate_for_fit=temporal_covariate_for_fit,
                                            remove_physically_implausible_models=remove_physically_implausible_models,
                                            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                            param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                            )
                nllh_value = xp.run_one_experiment(gcm_rcm_couple)
                update_csv(csv_filepath, combination_name, altitude, gcm_rcm_couple, nllh_value)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_model_as_truth_experiment()
