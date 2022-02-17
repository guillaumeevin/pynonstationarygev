import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_trend.ensemble_fit.visualizer_for_simple_case import VisualizerForSimpleCase
from projected_extremes.section_results.utils.get_nb_linear_pieces import run_selection
from projected_extremes.section_results.utils.setting_utils import set_up_and_load, get_last_year_for_the_train_set


def main_simple_visualizatoin():
    # Set parameters

    # fast = False considers all ensemble members and all elevations,
    # fast = None considers all ensemble members and 1 elevation,
    # fast = True considers only 6 ensemble mmebers and 1 elevation

    # snowfall=True corresponds to daily snowfall
    # snowfall=False corresponds to accumulated ground snow load
    # snowfall=None corresponds to daily winter precipitation
    fast = None
    snowfall = False
    with_bootstrap_interval = False

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    # For the article 3, we focus on 4 massifs for the elevation 1500
    altitudes = [1500]
    massif_names = ['Mont-Blanc', 'Ubaye', 'Champsaur', 'Vercors'][:]

    # Load the selected parameterization (adjustment coefficient and number of linear pieces)
    massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
        massif_names,
        altitudes[0],
        gcm_rcm_couples,
        safran_study_class,
        scenario,
        study_class,
        snowfall=snowfall,
        season=season)

    for massif_name, parametrization_number in massif_name_to_parametrization_number.items():

        model_classes = [massif_name_to_model_class[massif_name]]

        last_year_for_the_train_set = get_last_year_for_the_train_set(percentage=1)

        # Visualization is run only for massif where the selected parameterization is with adjustment
        if parametrization_number != 0:
            print('\n\n', massif_name)

            # Consider the parameterization without adjustment coefficients and the selected parameterization
            combinations = [(0, 0, 0), (parametrization_number, parametrization_number, 0)]

            # Run simple visualization to illustrate the probability density with and without the adjustment coefficients
            visualizer = VisualizerForSimpleCase(altitudes, gcm_rcm_couples, safran_study_class, study_class,
                                                 Season.annual,
                                                 scenario=scenario,
                                                 model_classes=model_classes,
                                                 massif_name=massif_name,
                                                 fit_method=fit_method,
                                                 temporal_covariate_for_fit=temporal_covariate_for_fit,
                                                 remove_physically_implausible_models=remove_physically_implausible_models,
                                                 display_only_model_that_pass_gof_test=False,
                                                 combinations_for_together=combinations,
                                                 linear_effects=linear_effects,
                                                 year_max_for_studies=None,
                                                 last_year_for_the_train_set=last_year_for_the_train_set,
                                                 with_bootstrap_interval=with_bootstrap_interval)
            visualizer.visualize_density_to_illustrate_adjustments(with_density=True)


if __name__ == '__main__':
    main_simple_visualizatoin()
