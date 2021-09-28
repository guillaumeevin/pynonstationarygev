import datetime
import os.path as op
import time

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel, \
    NonStationaryLocationAndScaleGumbelModel, NonStationaryLocationGumbelModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_simple_case import VisualizerForSimpleCase
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name_for_tuple, \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.get_nb_linear_pieces import run_selection
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from projects.projected_extreme_snowfall.results.setting_utils import get_last_year_for_the_train_set
from root_utils import get_display_name_from_object_type


def main_simple_visualizatoin():
    start = time.time()

    fast = True
    snowfall = False

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    altitudes = [1500]
    percentage = 1

    # Load the csv filepath
    massif_names = AbstractStudy.all_massif_names()[:1]
    gcm_rcm_couples = gcm_rcm_couples[:]
    massif_names = ['Thabor', 'Vanoise', 'Haute-Tarentaise'][:]
    massif_names, massif_name_to_model_class, massif_name_to_parametrization_number = run_selection(massif_names,
                                                                                      altitudes[0],
                                                                                                     gcm_rcm_couples,
                                                                                                     safran_study_class,
                                                                                                     scenario,
                                                                                                     study_class,
                                                                                      snowfall=snowfall)

    print(massif_name_to_parametrization_number)
    for massif_name, parametrization_number in massif_name_to_parametrization_number.items():

        model_classes = [massif_name_to_model_class[massif_name]]

        last_year_for_the_train_set = get_last_year_for_the_train_set(percentage)
        linear_effects = (False, False, False)

        display_only_model_that_pass_gof_test = False

        year_max_for_studies = None
        weight_on_observation = 1

        combinations = [(0, 0, 0)][:]
        if parametrization_number == 0:
            visualizer = VisualizerForSimpleCase(altitudes, gcm_rcm_couples, safran_study_class, study_class,
                                                 Season.annual,
                                                 scenario=scenario,
                                                 model_classes=model_classes,
                                                 massif_name=massif_name,
                                                 fit_method=fit_method,
                                                 temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                 remove_physically_implausible_models=remove_physically_implausible_models,
                                                 display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                 combinations_for_together=combinations,
                                                 weight_on_observation=weight_on_observation,
                                                 linear_effects=linear_effects,
                                                 year_max_for_studies=year_max_for_studies,
                                                 last_year_for_the_train_set=last_year_for_the_train_set,
                                                 )
            visualizer.visualize_density_to_illustrate_adjustments(with_density=False)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_simple_visualizatoin()
