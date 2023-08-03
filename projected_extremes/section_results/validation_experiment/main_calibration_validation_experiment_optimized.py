from projected_extremes.section_results.utils.get_nb_linear_pieces import get_massif_name_to_number
from projected_extremes.section_results.utils.selection_utils import number_to_model_class
from projected_extremes.section_results.utils.setting_utils import set_up_and_load, get_last_year_for_the_train_set
from projected_extremes.section_results.validation_experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment


def main_calibration_validation_experiment():
    # Set parameters

    # fast = False considers all ensemble members and all elevations,
    # fast = None considers all ensemble members and 1 elevation,
    # fast = True considers only 6 ensemble mmebers and 1 elevation

    # snowfall=True corresponds to daily snowfall
    # snowfall=False corresponds to accumulated ground snow load
    # snowfall=None corresponds to daily winter precipitation
    fast = False
    snowfall = True
    nb_days = 3

    # Load parameters
    altitudes_list, gcm_rcm_couples, all_massif_names, _, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall, nb_days)

    # altitudes_list = [[1500], [1800], [2100], [2400]]
    # all_massif_names = ['Mercantour']
    altitudes_list = [[2100], [2400], [2700], [3000], [3300], [3600]][:]

    # We consider three types of split where the training set represents either 60%, 70% or 80% of the reanalysis data
    percentage_for_the_train_set = [0.6, 0.7, 0.8][:]

    # Loop on the altitudes
    for altitudes in altitudes_list:
        altitude = altitudes[0]
        print('\n', altitudes)

        # Load the number of pieces for the piecewise linear functions selected with the model as truth experiment
        # linear effects = (False, False, False) means that the adjustment are constant for the three parameters of the GEV distributions
        # However if you set linear effects = (True, False, False) it would mean the the adjustment coefficient for the
        # first parameter of the GEV distribution (the location parameter) is changing lienarly with the global warming
        massif_name_to_number, linear_effects, massif_names, _, _, _ = get_massif_name_to_number(altitude,
                                                                                              gcm_rcm_couples,
                                                                                              all_massif_names,
                                                                                              safran_study_class,
                                                                                              scenario,
                                                                                              snowfall,
                                                                                              study_class,
                                                                                              season)
        # Loop on the massifs
        for massif_name, number in massif_name_to_number.items():
            print('\n', massif_name, number)

            model_classes = [number_to_model_class[number]]

            for percentage in percentage_for_the_train_set:

                # Set the last year for the train, and the first year for the test set (for the S2M reanalysis)
                last_year_for_the_train_set = get_last_year_for_the_train_set(percentage)
                start_year_for_the_test_set = last_year_for_the_train_set + 1
                print('Last year for the train set', last_year_for_the_train_set, 'Percentage', percentage)

                # Loop on the potential parameterization
                # 0 represents the parameterization without adjustment coefficients
                # 1, 2, 4, 5 represents four different parameterization with adjustment coefficients
                for i in [0, 1, 2, 4, 5][:]:
                    print("parameterization:", i)

                    # The line below states that:
                    # For the 2 first parameters of the GEV distribution (location and scale parameters)
                    # we potentially consider adjustment coefficients
                    # For the last parameter of the GEV distribution (the shape parameter)
                    # 0 means that we do not consider any adjustment coefficients
                    combination = (i, i, 0)

                    # Load a CalibrationValidationExperiment and write the result of this experiment in a csv file
                    xp = CalibrationValidationExperiment(altitudes, gcm_rcm_couples, safran_study_class, study_class,
                                                         season, scenario=scenario, selection_method_names=['aic'],
                                                         model_classes=model_classes, massif_names=[massif_name],
                                                         fit_method=fit_method,
                                                         temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                         remove_physically_implausible_models=remove_physically_implausible_models,
                                                         display_only_model_that_pass_gof_test=False,
                                                         combination=combination, linear_effects=linear_effects,
                                                         start_year_for_test_set=start_year_for_the_test_set,
                                                         year_max_for_studies=None)
                    xp.run_one_experiment()


if __name__ == '__main__':
    main_calibration_validation_experiment()
