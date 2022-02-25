import matplotlib

from extreme_trend.ensemble_fit.together_ensemble_fit.visualizer_non_stationary_ensemble import \
    VisualizerNonStationaryEnsemble

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import run_selection
from projected_extremes.section_results.utils.setting_utils import set_up_and_load

from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def quantitative(pvalues, visualizer):
    # Create an histogram for the metric
    ax = plt.gca()
    count_above_5_percent = [int(m >= 0.05) for m in pvalues]
    percentage_above_5_percent = 100 * sum(count_above_5_percent) / len(count_above_5_percent)
    print("Percentage above 5 percent", percentage_above_5_percent)
    ax.hist(pvalues, bins=20, range=[0, 1], weights=np.ones(len(pvalues)) / len(pvalues))
    ax.set_xlim((0, 1))
    ylim = ax.get_ylim()
    ax.vlines(0.05, ymin=ylim[0], ymax=ylim[1], color='k', linestyles='dashed', label='0.05 significance level')
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xlabel('p-value for the Anderson-Darling test')
    ax.set_ylabel('Percentage')
    ax.legend()
    visualizer.plot_name = 'All pvalues'
    visualizer.show_or_save_to_file()


def main():
    # Set parameters

    # fast = False considers all ensemble members and all elevations,
    # fast = None considers all ensemble members and 1 elevation,
    # fast = True considers only 6 ensemble mmebers and 1 elevation

    # snowfall=True corresponds to daily snowfall
    # snowfall=False corresponds to accumulated ground snow load
    # snowfall=None corresponds to daily winter precipitation
    fast = None
    snowfall = False

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, study_class, temporal_covariate_for_fit, \
    remove_physically_implausible_models, display_only_model_that_pass_gof_test, safran_study_class, fit_method, \
    season = set_up_and_load(fast, snowfall)

    # Loop on the altitudes
    for altitudes in altitudes_list:

        # Load the selected parameterization (adjustment coefficient and number of linear pieces)
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
            AbstractStudy.all_massif_names()[:],
            altitudes[0],
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
            season=season)

        massif_name_to_param_name_to_climate_coordinates_with_effects = {}
        for massif_name, parametrization_number in massif_name_to_parametrization_number.items():
            print('parameterization number for the effects:', parametrization_number)

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
            ensemble_fit_classes=[TogetherEnsembleFit],
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
        sub_visualizer = [together_ensemble_fit.visualizer
                          for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)][0] # type: VisualizerNonStationaryEnsemble

        all_pvalues = []
        for massif_name, one_fold_fit in sub_visualizer.massif_name_to_one_fold_fit.items():
            _, test_names, pvalues = one_fold_fit.goodness_of_fit_test_separated_for_each_gcm_rcm_couple(one_fold_fit.best_estimator)
            all_pvalues.extend(pvalues)

        print(len(all_pvalues), all_pvalues)
        quantitative(all_pvalues, sub_visualizer)

if __name__ == '__main__':
    main()
