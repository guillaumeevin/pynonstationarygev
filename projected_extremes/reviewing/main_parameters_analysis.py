import os.path as op

import matplotlib
import numpy as np
import pandas as pd

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import StationaryTemporalModel, \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_trend.one_fold_fit.utils import load_sub_visualizer
from projected_extremes.reviewing.parameter_plot import parameter_plot, get_parameter_values_list, shape_parameter_plot
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


def main_parameters():
    fast = False
    snowfall = True

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, study_class, temporal_covariate_for_fit, \
        remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class, fit_method, \
        season = set_up_and_load(fast, snowfall)

    # Loop on the altitudes
    # altitudes_list = [[1500]]
    altitudes_list = [[900], [1200], [1500], [1800], [2100], [2400], [2700], [3000], [3300], [3600]][:]

    # print(altitudes_list)
    # for mode in range(4):
    # for mode in range(6):

    parameter_values_list = []
    covariates = np.linspace(1.5, 4, num=50)

    for altitudes in altitudes_list[:]:

        altitude = altitudes[0]

        # Load the selected parameterization (adjustment coefficient and number of linear pieces)
        all_massif = True
        all_massif_names = AbstractStudy.all_massif_names()[:] if all_massif else ['Mont-Blanc']
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects, gcm_rcm_couple_to_studies = run_selection(
            all_massif_names,
            altitude,
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
            season=season, plot_selection_graph=False)




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

        sub_visualizer = load_sub_visualizer(altitudes, display_only_model_that_pass_gof_test, fit_method,
                                             gcm_rcm_couples, linear_effects, massif_name_to_model_class,
                                             massif_name_to_param_name_to_climate_coordinates_with_effects,
                                             massif_names, remove_physically_implausible_models,
                                             safran_study_class, scenario, season, study_class,
                                             temporal_covariate_for_fit, gcm_rcm_couple_to_studies)
        parameter_values_list.extend(get_parameter_values_list(sub_visualizer, covariates))
    # Create plot
    print(f'number of selected models {len(parameter_values_list)}')
    shape_parameter_plot(sub_visualizer, covariates, parameter_values_list)




if __name__ == '__main__':
    main_parameters()
