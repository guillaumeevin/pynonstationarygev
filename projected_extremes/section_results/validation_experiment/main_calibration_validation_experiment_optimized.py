import datetime
import time

from projected_extremes.section_results.utils.get_nb_linear_pieces import get_massif_name_to_number
from projected_extremes.section_results.utils.selection_utils import number_to_model_class
from projected_extremes.section_results.utils.setting_utils import set_up_and_load, get_last_year_for_the_train_set
from projected_extremes.section_results.validation_experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment


def main_calibration_validation_experiment():
    start = time.time()

    fast = True
    snowfall = True

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    # print('sleeping...')
    # time.sleep(60*60*4)

    # Load the csv filepath
    calibration_class = CalibrationValidationExperiment
    year_max_for_studies = None

    l = [0.6, 0.7, 0.8][:]

    for altitudes in altitudes_list:
        altitude = altitudes[0]
        print('\n', altitudes)
        massif_name_to_number, linear_effects, massif_names, _, _ = get_massif_name_to_number(altitude,
                                                                                                      gcm_rcm_couples,
                                                                                                      massif_names,
                                                                                                      safran_study_class,
                                                                                                      scenario,
                                                                                                      snowfall,
                                                                                                      study_class,
                                                                                              season)
        for massif_name, number in massif_name_to_number.items():
            print('\n', massif_name, number)
            model_classes = [number_to_model_class[number]]
            for percentage in l:
                # percentage += 0.6
                last_year_for_the_train_set = get_last_year_for_the_train_set(percentage)
                start_year_for_the_test_set = last_year_for_the_train_set + 1

                display_only_model_that_pass_gof_test = False

                print('Last year for the train set', last_year_for_the_train_set, 'Percentage', percentage)
                print('year max for studies:', year_max_for_studies)

                for i in [0, 1, 2, 4, 5][:]:
                    # for i in [-1, 0, 5][1:]:
                    print("parameterization:", i)
                    # combination = (i, i, 0)
                    combination = (i, i, 0)
                    xp = calibration_class(altitudes, gcm_rcm_couples, safran_study_class, study_class, season,
                                           scenario=scenario,
                                           selection_method_names=['aic'],
                                           model_classes=model_classes,
                                           massif_names=[massif_name],
                                           fit_method=fit_method,
                                           temporal_covariate_for_fit=temporal_covariate_for_fit,
                                           remove_physically_implausible_models=remove_physically_implausible_models,
                                           display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                           combination=combination,
                                           linear_effects=linear_effects,
                                           start_year_for_test_set=start_year_for_the_test_set,
                                           year_max_for_studies=year_max_for_studies)
                    xp.run_one_experiment()
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_calibration_validation_experiment()
