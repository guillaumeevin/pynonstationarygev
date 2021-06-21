import numpy as np

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_crocus import AdamontSnowLoad
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.adamont_data.cmip5.climate_explorer_cimp5 import year_to_averaged_global_mean_temp
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_max_swe import CrocusSnowLoad2019
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load
from root_utils import get_display_name_from_object_type


def simulation_parameters_from_study():
    """Fit a non-stationary GEV with a linear non-stationarity on a all the parameters"""
    fast = None
    snowfall = False
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)
    altitudes = altitudes_list[0]

    d = year_to_averaged_global_mean_temp(scenario)

    # Select the model with minimal AIC
    gcm_rcm_couple_to_studies = VisualizerForProjectionEnsemble.load_gcm_rcm_couple_to_studies(altitudes, gcm_rcm_couples,
                                                                                               None, safran_study_class,
                                                                                               scenario, Season.annual,
                                                                                               study_class)
    OneFoldFit.SELECTION_METHOD_NAME = 'aic'

    visualizer = VisualizerNonStationaryEnsemble(gcm_rcm_couple_to_studies, model_classes,
                                                          False, massif_names, fit_method,
                                                          temporal_covariate_for_fit,
                                                          False,
                                                          False,
                                                          remove_physically_implausible_models,
                                                          None)
    one_fold_fit = visualizer.massif_name_to_one_fold_fit['Vanoise']
    print(get_display_name_from_object_type(one_fold_fit.best_estimator.margin_model))
    margin_function = one_fold_fit.best_estimator.margin_function_from_fit
    print(margin_function.coef_dict)

    # coordinate_start = one_fold_fit.best_estimator.df_coordinates_temp.min().values
    coordinate_start = np.array([d[1951]])
    coordinate_end = np.array([d[2100]])
    print(coordinate_start, coordinate_end)
    # coordinate_end = one_fold_fit.best_estimator.df_coordinates_temp.max().values
    gev_params_start = margin_function.get_params(coordinate_start)
    gev_params_end = margin_function.get_params(coordinate_end)
    print(gev_params_start)
    print(gev_params_end)
    for param_value_start, param_value_end in zip(gev_params_start.param_values, gev_params_end.param_values):
        relative_difference = 100 * (param_value_end - param_value_start) / param_value_start
        print(relative_difference)


if __name__ == '__main__':
    simulation_parameters_from_study()