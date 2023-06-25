import os.path as op
import matplotlib
import pandas as pd

from projected_extremes.reviewing.reviewing_utils import load_csv_filepath_gof, load_parameters

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import run_selection
from projected_extremes.section_results.utils.setting_utils import set_up_and_load

from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble


def main():
    # Set parameters

    # fast = False considers all ensemble members and all elevations,
    # fast = None considers all ensemble members and 1 elevation,
    # fast = True considers only 6 ensemble mmebers and 1 elevation

    # snowfall=True corresponds to daily snowfall
    # snowfall=False corresponds to accumulated ground snow load
    # snowfall=None corresponds to daily winter precipitation
    fast = False
    snowfall = True

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, study_class, temporal_covariate_for_fit, \
    remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class, fit_method, \
    season = set_up_and_load(fast, snowfall)

    # Loop on the altitudes
    altitudes_list = [[2700]]
    print(altitudes_list)
    # for mode in range(4):
    # for mode in range(6):
    for mode in [8]:
        for altitudes in altitudes_list[:]:

            altitude = altitudes[0]

            # Load the selected parameterization (adjustment coefficient and number of linear pieces)
            all_massif = False
            all_massif_names = AbstractStudy.all_massif_names()[:] if all_massif else ['Mont-Blanc']
            massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
                all_massif_names,
                altitude,
                gcm_rcm_couples,
                safran_study_class,
                scenario,
                study_class,
                snowfall=snowfall,
                season=season)

            csv_filepath = load_csv_filepath_gof(mode, altitude, all_massif)
            massif_name_to_model_class, massif_name_to_parametrization_number \
                = load_parameters(mode, massif_name_to_model_class, massif_name_to_parametrization_number)

            csv_filename = op.basename(csv_filepath)
            if op.exists(csv_filepath):
                print('already done: {}'.format(csv_filename))
            else:
                print('run: {}'.format(csv_filename))


                massif_name_to_param_name_to_climate_coordinates_with_effects = {}
                for massif_name, parametrization_number in massif_name_to_parametrization_number.items():
                    # print('parameterization number for the effects:', parametrization_number)

                    # The line below states that:

                    # For the 2 first parameters of the GEV distribution (location and scale parameters)
                    # we potentially consider adjustment coefficients defined by the parameterization number
                    # 0 represents the parameterization without adjustment coefficients
                    # 1, 2, 4, 5 represents four different parameterization with adjustment coefficients

                    # For the last parameter of the GEV distribution (the shape parameter)
                    # 0 means that we do not consider any adjustment coefficients
                    combination = (parametrization_number, parametrization_number, 0)

                    # Set the selected parameterization of adjustment coefficient for each massif
                    param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                        combination)
                    massif_name_to_param_name_to_climate_coordinates_with_effects[
                        massif_name] = param_name_to_climate_coordinates_with_effects

                # Visualize together the values for all massifs on a map
                visualizer = VisualizerForProjectionEnsemble(
                    [altitudes], gcm_rcm_couples, study_class, season, scenario,
                    model_classes=massif_name_to_model_class,
                    massif_names=massif_names,
                    fit_method=fit_method,
                    temporal_covariate_for_fit=temporal_covariate_for_fit,
                    remove_physically_implausible_models=remove_physically_implausible_models,
                    safran_study_class=safran_study_class,
                    linear_effects=linear_effects,
                    display_only_model_that_pass_gof_test=False,
                    param_name_to_climate_coordinates_with_effects=massif_name_to_param_name_to_climate_coordinates_with_effects,
                )

                with_significance = False
                sub_visualizer = visualizer.visualizer

                all_pvalues = []
                for massif_name, one_fold_fit in sub_visualizer.massif_name_to_one_fold_fit.items():
                    _, test_names, pvalues = one_fold_fit.goodness_of_fit_test_separated_for_each_gcm_rcm_couple(one_fold_fit.best_estimator)
                    all_pvalues.extend(pvalues)

                # Save values to csv
                pd.Series(all_pvalues).to_csv(csv_filepath)



if __name__ == '__main__':
    main()
