import datetime
import time

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleTemporalModel, NonStationaryLocationAndScaleGumbelModel, \
    NonStationaryLocationGumbelModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projected_extremes.results.experiment import \
    CalibrationValidationExperiment
from projected_extremes.results.experiment import \
    CalibrationAicExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from projects.projected_extreme_snowfall.results.setting_utils import get_last_year_for_the_train_set


def main_calibration_validation_experiment():
    start = time.time()

    fast = True
    snowfall = True

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    # print('sleeping...')
    # time.sleep(60*40)

    # Load the csv filepath
    altitudes_list = [[2100]]
    calibration_class = CalibrationAicExperiment
    year_max_for_studies = None
    linear_effects = (False, False, False)


    display_only_model_that_pass_gof_test = False

    # weight_on_observation = 1 + 20
    # weight_on_observation = 1 + 9
    weight_on_observation = 1
    # weight_on_observation = 1 + 20*13
    print('weight on observation=', weight_on_observation)

    for massif_name in massif_names:
        print(massif_name)
        for altitudes in altitudes_list:
            for i in [-1, 0, 1, 2, 4, 5][:]:
            # for i in [-1, 0, 5][1:]:
                print("parameterization:", i)
                # combination = (i, i, 0)
                combination = (i, i, 0)
                xp = calibration_class(altitudes, gcm_rcm_couples, safran_study_class, study_class, Season.annual,
                                       scenario=scenario,
                                       selection_method_names=['aic'],
                                       model_classes=model_classes,
                                       massif_names=[massif_name],
                                       fit_method=fit_method,
                                       temporal_covariate_for_fit=temporal_covariate_for_fit,
                                       remove_physically_implausible_models=remove_physically_implausible_models,
                                       display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                       combination=combination,
                                       weight_on_observation=weight_on_observation,
                                       linear_effects=linear_effects,
                                       year_max_for_studies=year_max_for_studies,
                                       )
                xp.run_one_experiment()
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_calibration_validation_experiment()
