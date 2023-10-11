import matplotlib

from extreme_trend.one_fold_fit.excel_from_one_fold_fit import to_excel
from extreme_trend.one_fold_fit.utils import load_sub_visualizer

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import run_selection
from projected_extremes.section_results.utils.setting_utils import set_up_and_load



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
    nb_days = 3

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, study_class, temporal_covariate_for_fit, \
    remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class, fit_method, \
    season = set_up_and_load(fast, snowfall, nb_days)

    altitudes_list = [[1500], [1800], [2100], [2400], [2700]][:]

    # Loop on the altitudes
    for altitudes in altitudes_list:

        # Load the selected parameterization (adjustment coefficient and number of linear pieces)
        massif_names = [ 'Mercantour']
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects, gcm_rcm_couple_to_studies = run_selection(
            massif_names,
            altitudes[0],
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
            season=season,
        plot_selection_graph=False)

        massif_name_to_param_name_to_climate_coordinates_with_effects = {}
        for massif_name, parametrization_number in massif_name_to_parametrization_number.items():

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


        for one_fold_fit in sub_visualizer.massif_name_to_one_fold_fit.values():
            to_excel(one_fold_fit, gcm_rcm_couple_to_studies)


if __name__ == '__main__':
    main()
