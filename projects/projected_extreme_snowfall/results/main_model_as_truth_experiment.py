import os.path as op
import time

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel, NonStationaryLocationAndScaleGumbelModel, \
    NonStationaryLocationAndScaleTemporalModel, StationaryTemporalModel
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel, NonStationaryThreeLinearLocationAndScaleAndShapeModel, \
    NonStationaryFourLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects, load_combination_name_for_tuple
from projects.projected_extreme_snowfall.results.experiment.model_as_truth_experiment import ModelAsTruthExperiment
from projects.projected_extreme_snowfall.results.part_2.average_bias import plot_average_bias, load_study, \
    compute_average_bias, plot_bias, plot_time_series
from projects.projected_extreme_snowfall.results.part_2.v1.main_mas_v1 import CSV_PATH
from projects.projected_extreme_snowfall.results.part_2.v2.utils import update_csv, is_already_done, load_excel, \
    main_sheet_name
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load, get_last_year_for_the_train_set


def main_preliminary_projections():
    # Load parameters
    show = False
    # print('sleeping...')
    # time.sleep(60*30)

    fast = False
    snowfall = None

    if show in [None, True]:
        matplotlib.use('Agg')
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

    year_max_for_pseudo_obs, year_max_for_gcm = 2019, 2100


    # for percentage in [0.7]:
    # year_max_for_pseudo_obs, year_max_for_gcm = get_last_year_for_the_train_set(percentage), 2019
    weight_on_observation = 1
    print('weight on observation=', weight_on_observation)

    linear_effects = (False, False, False)

    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    model_classes_list = [StationaryTemporalModel,
                          NonStationaryLocationAndScaleAndShapeTemporalModel,
                          NonStationaryTwoLinearLocationAndScaleAndShapeModel,
                          NonStationaryThreeLinearLocationAndScaleAndShapeModel,
                          NonStationaryFourLinearLocationAndScaleAndShapeModel][:]
    altitudes_list = [[900], [1500], [2100], [2700], [3300]]

    for model_class in model_classes_list:
        model_classes = [model_class]
        for altitudes in altitudes_list:
            run_mas(altitudes, display_only_model_that_pass_gof_test, fast, gcm_rcm_couples, massif_names,
                    model_classes, remove_physically_implausible_models, safran_study_class, scenario, show, snowfall,
                    study_class, temporal_covariate_for_fit, year_max_for_gcm, year_max_for_pseudo_obs,
                    weight_on_observation, linear_effects, fit_method)


def run_mas(altitudes, display_only_model_that_pass_gof_test, fast, gcm_rcm_couples, massif_names,
            model_classes, remove_physically_implausible_models, safran_study_class, scenario, show, snowfall,
            study_class, temporal_covariate_for_fit, year_max_for_gcm, year_max_for_pseudo_obs, weight_on_observation,
            linear_effects, fit_method):
    altitude = altitudes[0]
    print('Altitude={}'.format(altitude))
    gcm_rcm_couple_to_study, safran_study = load_study(altitude, gcm_rcm_couples, safran_study_class, scenario,
                                                       study_class)

    for massif_name in massif_names[::1]:
        if massif_name in safran_study.study_massif_names:
            print(massif_name)

            average_bias, _ = compute_average_bias(gcm_rcm_couple_to_study, massif_name, safran_study, show=show)
            gcm_rcm_couples_sampled_for_experiment, gcm_rcm_couple_to_average_bias, gcm_rcm_couple_to_gcm_rcm_couple_to_biases = plot_average_bias(
                gcm_rcm_couple_to_study, massif_name, average_bias,
                alpha=1000, show=show)

            print("Number of couples:", len(gcm_rcm_couples_sampled_for_experiment))
            for i in [-1, 0, 1, 2, 4, 5][1:2]:
                # for i in [-1, 0, 5][:]:
                print('parametrization', i)
                for gcm_rcm_couple in gcm_rcm_couples_sampled_for_experiment:
                    combination = (i, i, 0)

                    xp = ModelAsTruthExperiment(altitudes, gcm_rcm_couples,
                                                safran_study_class,
                                                study_class, Season.annual,
                                                scenario=scenario,
                                                selection_method_names=['aic'],
                                                model_classes=model_classes,
                                                massif_names=[massif_name],
                                                fit_method=fit_method,
                                                temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                remove_physically_implausible_models=remove_physically_implausible_models,
                                                display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
                                                gcm_rcm_couples_sampled_for_experiment=gcm_rcm_couples_sampled_for_experiment,
                                                combination=combination,
                                                year_max_for_gcm=year_max_for_gcm,
                                                year_max_for_pseudo_obs=year_max_for_pseudo_obs,
                                                weight_on_observation=weight_on_observation,
                                                linear_effects=linear_effects,
                                                )
                    xp.run_one_experiment(gcm_rcm_couple_as_pseudo_truth=gcm_rcm_couple)


if __name__ == '__main__':
    main_preliminary_projections()
