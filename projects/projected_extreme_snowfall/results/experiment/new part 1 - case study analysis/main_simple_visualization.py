import datetime
import os.path as op
import time

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel, \
    NonStationaryLocationAndScaleGumbelModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_simple_case import VisualizerForSimpleCase
from projects.projected_extreme_snowfall.results.combination_utils import load_combination_name_for_tuple, \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.experiment.calibration_validation_experiment import \
    CalibrationValidationExperiment
from projects.projected_extreme_snowfall.results.part_3.main_projections_ensemble import set_up_and_load
from projects.projected_extreme_snowfall.results.setting_utils import get_last_year_for_the_train_set
from root_utils import get_display_name_from_object_type

def main_simple_visualizatoin():
    start = time.time()

    fast = False
    snowfall = False

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    # Load the csv filepath
    massif_names = AbstractStudy.all_massif_names()
    # massif_names = ['Chartreuse']

    # Good fits
    # massif_names = ['Haute-Maurienne', 'Thabor', 'Queyras']

    # Bad fits
    # massif_names = ['Devoluy', 'Mercantour']
    massif_names = ['Devoluy', 'Vercors']

    for massif_name in massif_names:
        # indexes = [5, 8, 9, 13, 18]
        # indexes = [1, 5, 6, 7, 8, 14, 15, 17, 18, 16]
        # indexes = list(set(range(20))-set(indexes))
        # gcm_rcm_couples = [gcm_rcm_couples[i] for i in indexes]
        gcm_rcm_couples = gcm_rcm_couples[:]
        altitudes = [1500]
        percentage = 0.7
        fit_method = MarginFitMethod.extremes_fevd_mle
        model_classes = [NonStationaryLocationAndScaleTemporalModel]
        model_classes = [NonStationaryLocationAndScaleGumbelModel]
        last_year_for_the_train_set = get_last_year_for_the_train_set(percentage)
        linear_effects = (False, False, False)

        display_only_model_that_pass_gof_test = False

        print('Last year for the train set', last_year_for_the_train_set, 'Percentage', percentage)
        year_max_for_studies = 2019
        print('year max for studies:', year_max_for_studies)
        # weight_on_observation = 1 + 20
        weight_on_observation = 1
        print('weight on observation=', weight_on_observation)

        combinations = [(0, 0, 0)][:]
        for i in [5, 1, 2, 4]:
            combinations.append((i, 0, 0))
        # combinations.append((5,5,5))

        visualizer = VisualizerForSimpleCase(altitudes, gcm_rcm_couples, safran_study_class, study_class, Season.annual,
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
        visualizer.visualize_gev_parameters()
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_simple_visualizatoin()
