import matplotlib as mpl

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from collections import OrderedDict
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import get_min_max_number_of_pieces, run_selection
from projected_extremes.section_results.utils.print_table_model_selected import print_table_model_selected
from projected_extremes.section_results.utils.projection_elevation_plot_utils import plot_piechart_scatter_plot, \
    plot_contour_changes_values, plot_transition_lines, plot_relative_change_at_massif_level
from projected_extremes.section_results.utils.selection_utils import short_name_to_parametrization_number, \
    model_class_to_number
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
    fast = None
    snowfall = True

    # Load parameters
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    all_massif_names = AbstractStudy.all_massif_names()[:]

    altitudes = [e[0] for e in altitudes_list]

    # Initialize a dataframe called df_model_selected to analyze the repartition of selected models (repartition of
    # the selected number of linear pieces, repartition of the selected parameterization of adjustment coefficients)
    parameterization_numbers = sorted(list(short_name_to_parametrization_number.values()))
    max_number, min_number, _ = get_min_max_number_of_pieces(snowfall)
    df_model_selected = pd.DataFrame(0, index=parameterization_numbers, columns=list(range(min_number, max_number + 1)))

    visualizers = []
    for altitude in altitudes:
        print('altitude', altitude)
        altitudes_list = [[altitude]]

        # Load the selected parameterization (adjustment coefficient and number of linear pieces)
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(
            all_massif_names,
            altitude,
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
            season=season)

        # Fill the dataframe
        for massif_name in massif_names:
            df_model_selected.loc[massif_name_to_parametrization_number[massif_name],
                                  model_class_to_number[massif_name_to_model_class[massif_name]]] += 1

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

        # Load and add a visualizer to the list of visualizers
        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, season, scenario,
            model_classes=massif_name_to_model_class,
            ensemble_fit_classes=[TogetherEnsembleFit],
            massif_names=massif_names,
            fit_method=fit_method,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            safran_study_class=safran_study_class,
            linear_effects=linear_effects,
            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
            param_name_to_climate_coordinates_with_effects=massif_name_to_param_name_to_climate_coordinates_with_effects,
        )
        sub_visualizer = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)][0]
        visualizers.append(sub_visualizer)

    return_periods = [None, 2, 5, 10, 20, 50, 100]
    print_table_model_selected(df_model_selected)

    if snowfall is True:
        elevations_for_contour_plot = [2100, 2400, 2700, 3000, 3300, 3600]
        visualizers_for_contour_plot = [v for v in visualizers if v.study.altitude in elevations_for_contour_plot]
        if len(visualizers_for_contour_plot) > 0:

            # Illustrate the percentage of massifs
            covariates = [1.5, 2, 2.5, 3, 3.5, 4][:]

            # Visualize the evolution of the relative change in return levels with global warming
            relative_change = True

            # Visualize the distribution of trends in return levels for the return level of interest
            # and for the mean annual maxima (which correspond to return period = None)
            for return_period in [OneFoldFit.return_period, None]:
                plot_piechart_scatter_plot(visualizers_for_contour_plot, all_massif_names, covariates,
                                           relative_change,
                                           return_period, snowfall)

            # Illustrate the contour with all elevation
            return_period_to_paths = OrderedDict()
            for return_period in return_periods[:]:
                paths = plot_contour_changes_values(visualizers_for_contour_plot, relative_change, return_period,
                                                    snowfall)
                return_period_to_paths[return_period] = paths

            # Plot transition line together
            for return_periods_for_plots in [[None, 2, 5, 10, 20, 50, 100], [100, 200, 500, 1000, 2000]][:1]:
                local_return_period_to_paths = OrderedDict()
                for r in return_periods_for_plots:
                    local_return_period_to_paths[r] = return_period_to_paths[r]
                plot_transition_lines(visualizers[0], local_return_period_to_paths, relative_change)

    if snowfall:
        # For snowfall, we visualize each massif and the average value on all massifs
        all_massif_names += [None]
    else:
        # For snow load and winter precipitation, we only visualize the value on all massifs
        all_massif_names = [None]

    # Illustrate the trend of each massif (and also for all massifs which correspond to massif_name = None)
    with_significance = False

    # Visualize the relative changes and the absolute changes
    for relative_change in [True, False][:]:

        # Loop on each massif
        for massif_name in all_massif_names:

            # For each return period, we specify the elevations that have the same trend
            # (decreasing, increasing then decreasing, increasing) to visualize these elevations together
            if snowfall is True:
                return_periods_for_plots = [return_periods[0], return_periods[-1]]
                return_period_to_categories_list = {
                    return_periods[-1]: [[900, 1200, 1500, 1800, 2100, 2400], [2700, 3000], [3300, 3600]],
                    return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000], [3300], [3600]],
                }
            elif snowfall is None:
                return_periods_for_plots = [return_periods[0], return_periods[-1]]
                return_period_to_categories_list = {
                    return_periods[-1]: [[900, 1200, 1500, 1800, 2100], [2400, 2700, 3000, 3300, 3600]],
                    return_periods[0]: [[900, 1200, 1500, 1800, 2100], [2400, 2700, 3000, 3300, 3600]],
                }
            else:
                return_periods_for_plots = [return_periods[-2]]
                return_period_to_categories_list = {
                    return_periods[-2]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]]
                }

            # Visualize the evolution of these changes
            for return_period in return_periods_for_plots:
                categories_list = return_period_to_categories_list[return_period]
                for categories in categories_list:
                    categories = set(categories)
                    visualizers_categories = [v for v in visualizers if v.study.altitude in categories]
                    if len(visualizers_categories) > 0:
                        for temperature_covariate in [True, False]:
                            plot_relative_change_at_massif_level(visualizers_categories, massif_name, True,
                                                                 relative_change, return_period,
                                                                 snowfall, temperature_covariate)


if __name__ == '__main__':
    main()
