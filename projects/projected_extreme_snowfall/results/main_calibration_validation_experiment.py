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
    CalibrationValidationExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from root_utils import get_display_name_from_object_type

gcm_couple_fake = ("", "")


def main_calibration_validation_experiment():
    start = time.time()

    fast = None
    snowfall = None

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    # Load the csv filepath
    percentage = 0.85
    last_year_for_the_train_set = 1959 + round(percentage*61) - 1
    start_year_for_the_test_set = last_year_for_the_train_set + 1
    print(percentage, start_year_for_the_test_set)
    display_only_model_that_pass_gof_test = True
    # massif_names = ['Mont-Blanc']

    year_max_for_studies = 2019
    print('year max for studies:', year_max_for_studies)

    for massif_name in massif_names:
        for altitudes in altitudes_list:
            for i in [0, 1, 2, 4, 5]:
                print(i)
                combination = (i, i, 0)
                xp = CalibrationValidationExperiment(altitudes, gcm_rcm_couples, safran_study_class, study_class, Season.annual,
                                                     scenario=scenario,
                                                     selection_method_names=['aic'],
                                                     model_classes=model_classes,
                                                     massif_names=[massif_name],
                                                     fit_method=fit_method,
                                                     temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                     remove_physically_implausible_models=remove_physically_implausible_models,
                                                     display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                     param_name_to_climate_coordinates_with_effects=load_param_name_to_climate_coordinates_with_effects(combination),
                                                     start_year_for_test_set=start_year_for_the_test_set,
                                                     year_max_for_studies=year_max_for_studies)
                xp.run_one_experiment()
        end = time.time()
        duration = str(datetime.timedelta(seconds=end - start))
        print('Total duration', duration)


if __name__ == '__main__':
    main_calibration_validation_experiment()
