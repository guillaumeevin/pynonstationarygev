import datetime
import os.path as op
import time
from itertools import product

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name_for_tuple, \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.part_1.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.part_1.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidaitonExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load

gcm_couple_fake = ("", "")


def main_calibration_validation_experiment():
    start = time.time()

    fast = False
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class = set_up_and_load \
        (fast)

    # Load the csv filepath
    altitudes_str = '_'.join([str(a[0]) for a in altitudes_list])
    start_year_for_the_test_set = 2015

    csv_filename = 'fast_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_splityear_{}.csv'.format(fast, altitudes_str,
                                                                                                         len(model_classes),
                                                                                                         len(gcm_rcm_couples),
                                                                                           start_year_for_the_test_set)
    csv_filepath = op.join(CSV_PATH, csv_filename)

    # combinations = [(i, i, 0) for i in [3]]
    # combinations = [(i, i, 0) for i in range(4)]
    potential_indices = list(range(4))
    idx = 3
    combinations = [(c, idx, 0) for c in potential_indices]

    for altitudes in altitudes_list:
        altitude = altitudes[0]

        for i, combination in enumerate(combinations, 1):
            print("Running:", altitude,
                  'Combination {}/{}, {}'.format(i, len(combinations), combination))

            combination_name = load_combination_name_for_tuple(combination)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)

            if is_already_done(csv_filepath, combination_name, altitude, gcm_couple_fake):
                continue

            xp = CalibrationValidaitonExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                                 scenario=scenario,
                                                 selection_method_names=['aic'],
                                                 model_classes=model_classes,
                                                 massif_names=massif_names,
                                                 fit_method=MarginFitMethod.evgam,
                                                 temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                 remove_physically_implausible_models=remove_physically_implausible_models,
                                                 display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                 param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                 safran_study_class=safran_study_class,
                                                 start_year_for_test_set=start_year_for_the_test_set
                                                 )
            nllh_value = xp.run_one_experiment()
            update_csv(csv_filepath, combination_name, altitude, gcm_couple_fake, nllh_value)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_calibration_validation_experiment()
