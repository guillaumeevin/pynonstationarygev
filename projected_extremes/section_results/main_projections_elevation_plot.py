import matplotlib as mpl

from extreme_trend.one_fold_fit.utils import load_sub_visualizer

mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from collections import OrderedDict
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from projected_extremes.section_results.utils.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import get_min_max_number_of_pieces, run_selection
from projected_extremes.section_results.utils.print_table_model_selected import print_table_model_selected
from projected_extremes.section_results.utils.projection_elevation_plot_utils import plot_contour_changes_values, plot_transition_lines, plot_relative_change_at_massif_level
from projected_extremes.section_results.utils.selection_utils import short_name_to_parametrization_number, \
    model_class_to_number
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
    altitudes_list, gcm_rcm_couples, massif_names, _, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall, nb_days)

    all_massif_names = AbstractStudy.all_massif_names()[:]
    altitudes = [e[0] for e in altitudes_list][:]
    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][:]
    print(altitudes)
    # Quick mode
    # all_massif_names = ['Chartreuse', 'Bauges', 'Mont-Blanc','Queyras'][2:]

    # Initialize a dataframe called df_model_selected to analyze the repartition of selected models (repartition of
    # the selected number of linear pieces, repartition of the selected parameterization of adjustment coefficients)
    parameterization_numbers = sorted(list(short_name_to_parametrization_number.values()))
    max_number, min_number = get_min_max_number_of_pieces()
    df_model_selected = pd.DataFrame(0, index=parameterization_numbers, columns=list(range(min_number, max_number + 1)))

    visualizers = []
    for altitude in altitudes:
        print('altitude', altitude)

        # Load the selected parameterization (adjustment coefficient and number of linear pieces)
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects, gcm_rcm_couple_to_studies = run_selection(
            all_massif_names,
            altitude,
            gcm_rcm_couples,
            safran_study_class,
            scenario,
            study_class,
            snowfall=snowfall,
            season=season,
            plot_selection_graph=False)

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
        sub_visualizer = load_sub_visualizer(altitudes, display_only_model_that_pass_gof_test, fit_method,
                                             gcm_rcm_couples, linear_effects, massif_name_to_model_class,
                                             massif_name_to_param_name_to_climate_coordinates_with_effects,
                                             massif_names, remove_physically_implausible_models,
                                             safran_study_class, scenario, season, study_class,
                                             temporal_covariate_for_fit, gcm_rcm_couple_to_studies)
        visualizers.append(sub_visualizer)

    return_periods = [None, 2, 5, 10, 20, 50, 100]
    print_table_model_selected(df_model_selected)

    legend_fontsize = 16
    ticksize = 14

    if snowfall is True:

        # Visualize the evolution of the relative change in return levels with global warming
        relative_change = True

        # Illustrate the contour with all elevation
        return_period_to_paths = OrderedDict()
        elevations_for_contour_plot = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]
        visualizers_for_contour_plot = [v for v in visualizers if v.study.altitude in elevations_for_contour_plot]
        for return_period in return_periods[:]:
            paths = plot_contour_changes_values(visualizers_for_contour_plot, relative_change, return_period,
                                                snowfall, legend_fontsize, ticksize)
            return_period_to_paths[return_period] = paths

        # Plot transition line together
        for return_periods_for_plots in [[None, 2, 5, 10, 20, 50, 100], [100, 200, 500, 1000, 2000]][:1]:
            local_return_period_to_paths = OrderedDict()
            for r in return_periods_for_plots:
                local_return_period_to_paths[r] = return_period_to_paths[r]
            plot_transition_lines(visualizers[0], local_return_period_to_paths, relative_change, legend_fontsize,
                                  ticksize)

    if snowfall:
        # For snowfall, we visualize each massif and the average value on all massifs
        # all_massif_names += [None]
        all_massif_names = [None]
    else:
        # For snow load and winter precipitation, we only visualize the value on all massifs
        all_massif_names = [None]

    # Visualize the relative changes and the absolute changes
    for relative_change in [True, False][:]:

        # Loop on each massif
        for massif_name in all_massif_names:

            # For each return period, we specify the elevations that have the same trend
            # (decreasing, increasing then decreasing, increasing) to visualize these elevations together
            return_period_to_categories_list_color = None
            if snowfall is True:
                return_periods_for_plots = [return_periods[0], return_periods[-1]]
                return_period_to_categories_list = {
                    r: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]]
                    for r in return_periods_for_plots
                }
                if nb_days == 1:
                    return_period_to_categories_list_color = {
                        return_periods[-1]: [[900, 1200, 1500, 1800, 2100, 2400], [2700, 3000], [3300, 3600]],
                        return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000], [3300], [3600]],
                    }
                elif nb_days == 3:
                    return_period_to_categories_list_color = {
                        return_periods[-1]: [[900, 1200, 1500, 1800, 2100], [2400, 2700, 3000], [3300, 3600]],
                        return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700], [3000, 3300], [3600]],
                    }
                elif nb_days == 5:
                    return_period_to_categories_list_color = {
                        return_periods[-1]: [[900, 1200, 1500, 1800, 2100, 2400, 2700], [3000], [3300, 3600]],
                        return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700], [3000, 3300], [3600]],
                    }
                else:
                    raise NotImplementedError
            elif snowfall is None:
                return_periods_for_plots = [return_periods[0], return_periods[-1]]
                # return_periods_for_plots = [return_periods[-1]]
                return_period_to_categories_list = {
                    return_periods[-1]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]],
                    return_periods[0]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]],
                }
            else:
                return_periods_for_plots = [return_periods[-2]]
                return_period_to_categories_list = {
                    return_periods[-2]: [[900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]]
                }

            # Visualize the evolution of these changes
            for return_period in return_periods_for_plots:
                categories_list_color = None if return_period_to_categories_list_color is None else return_period_to_categories_list_color[return_period]
                categories_list = return_period_to_categories_list[return_period]
                for categories in categories_list:
                    categories = set(categories)
                    visualizers_categories = [v for v in visualizers if v.study.altitude in categories]
                    if len(visualizers_categories) > 0:
                        for temperature_covariate in [True, False][:1]:
                            plot_relative_change_at_massif_level(visualizers_categories, massif_name, True,
                                                                 relative_change, return_period,
                                                                 snowfall, temperature_covariate,
                                                                 categories_list_color, legend_fontsize,
                                                                 ticksize, nb_days)


if __name__ == '__main__':
    main()
