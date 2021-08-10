import datetime
import os.path as op
import time

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name_for_tuple, \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidaitonExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from root_utils import get_display_name_from_object_type

gcm_couple_fake = ("", "")


def main_calibration_validation_experiment():
    start = time.time()

    fast = False
    snowfall = False

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    # model_classes = [StationaryTemporalModel]
    model_classes = [NonStationaryLocationAndScaleAndShapeTemporalModel]
    # model_classes = [NonStationaryTwoLinearLocationAndScaleAndShapeModel]

    altitudes_list = [[2100]]
    # Load the csv filepath
    altitudes_str = '_'.join([str(a[0]) for a in altitudes_list])
    percentage = 0.85
    last_year_for_the_train_set = 1959 + round(percentage*61) - 1
    start_year_for_the_test_set = last_year_for_the_train_set + 1
    print(percentage, start_year_for_the_test_set)

    year_max_for_studies = None
    print('year max for studies:', year_max_for_studies)

    all_massif_names = AbstractStudy.all_massif_names()[::1]
    if fast:
        all_massif_names = all_massif_names[:2]
    for massif_name in all_massif_names:
        print(massif_name)
        massif_names = [massif_name]
        study = 'snowfall' if snowfall else 'snow_load'
        model_name = get_display_name_from_object_type(model_classes[0])
        csv_filename = 'fast_{}_{}_{}_altitudes_{}_nb_of_models_{}_nb_gcm_rcm_couples_{}_splityear_{}_year_max_studies_{}.xlsx'.format(fast,
                                                                                                                                       study,
                                                                                                                                       model_name,
                                                                                                             altitudes_str,
                                                                                                             len(model_classes),
                                                                                                             len(gcm_rcm_couples),
                                                                                                             start_year_for_the_test_set,
                                                                                                                                 year_max_for_studies)
        csv_filepath = op.join(CSV_PATH, csv_filename)

        for altitudes in altitudes_list:
            altitude = altitudes[0]

            for i in list(range(5))[:]:
                print(i)
                combination = (i, i, 0)

                combination_name = load_combination_name_for_tuple(combination)
                param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                    combination)

                experiment_name = str(altitude) + massif_name
                if is_already_done(csv_filepath, combination_name, experiment_name, gcm_couple_fake):
                    continue

                xp = CalibrationValidaitonExperiment(altitudes, gcm_rcm_couples, study_class, Season.annual,
                                                     scenario=scenario,
                                                     selection_method_names=['aic'],
                                                     model_classes=model_classes,
                                                     massif_names=massif_names,
                                                     fit_method=fit_method,
                                                     temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                     remove_physically_implausible_models=remove_physically_implausible_models,
                                                     display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                     param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
                                                     safran_study_class=safran_study_class,
                                                     start_year_for_test_set=start_year_for_the_test_set
                                                     )
                nllh_value = xp.run_one_experiment()
                update_csv(csv_filepath, combination_name, experiment_name, gcm_couple_fake, nllh_value)
        end = time.time()
        duration = str(datetime.timedelta(seconds=end - start))
        print('Total duration', duration)


if __name__ == '__main__':
    main_calibration_validation_experiment()
