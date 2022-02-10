import datetime
import time
from collections import OrderedDict

import matplotlib
import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projected_extremes.section_results.utils.combination_utils import load_param_name_to_climate_coordinates_with_effects
from projected_extremes.section_results.utils.get_nb_linear_pieces import get_min_max_number_of_pieces, run_selection
from projected_extremes.section_results.utils.print_table_model_selected import print_table_model_selected
from projected_extremes.section_results.utils.projection_elevation_plot_utils import plot_piechart_scatter_plot, \
    plot_contour_changes_values, plot_transition_lines, plot_relative_change_at_massif_level
from projected_extremes.section_results.utils.selection_utils import short_name_to_parametrization_number, \
    model_class_to_number
from projected_extremes.section_results.utils.setting_utils import set_up_and_load

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble


def main():
    start = time.time()

    fast = True
    snowfall = True
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    altitudes = [900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600][-4:]

    altitudes_list = [[900], [1200], [1500], [1800]][3:4]


    all_massif_names = AbstractStudy.all_massif_names()[:]

    if fast:
        altitudes = [1800]
        all_massif_names = ["Vanoise"]

    if fast is None:
        altitudes = altitudes[-2:]
        all_massif_names = ["Mont-Blanc", "Vanoise", "Oisans", "Grandes-Rousses"]

    parameterization_numbers = sorted(list(short_name_to_parametrization_number.values()))
    max_number, min_number, _ = get_min_max_number_of_pieces(snowfall)
    pieces_numbers = list(range(min_number, max_number + 1))
    df_model_selected = pd.DataFrame(0, index=parameterization_numbers, columns=pieces_numbers)
    visualizers = []
    for altitude in altitudes:
        print('altitude', altitude)
        altitudes_list = [[altitude]]

        ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]

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
            combination = (parametrization_number, parametrization_number, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(
                combination)
            massif_name_to_param_name_to_climate_coordinates_with_effects[
                massif_name] = param_name_to_climate_coordinates_with_effects

        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, season, scenario,
            model_classes=massif_name_to_model_class,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            fit_method=fit_method,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            safran_study_class=safran_study_class,
            linear_effects=linear_effects,
            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
            param_name_to_climate_coordinates_with_effects=massif_name_to_param_name_to_climate_coordinates_with_effects,
        )

        sub_visualizers = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)]
        print(len(sub_visualizers))
        sub_visualizer = sub_visualizers[0]
        visualizers.append(sub_visualizer)

    return_periods = [None, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000][:-4]
    print_table_model_selected(df_model_selected)

    if snowfall is True:
        elevations_for_contour_plot = [2100, 2400, 2700, 3000, 3300, 3600]
        visualizers_for_contour_plot = [v for v in visualizers if v.study.altitude in elevations_for_contour_plot]

        # Illustrate the percentage of massifs
        covariates = [1.5, 2, 2.5, 3, 3.5, 4][:]

        for relative_change in [True, False][:1]:
            for return_period in [OneFoldFit.return_period, None]:
                plot_piechart_scatter_plot(visualizers_for_contour_plot, all_massif_names, covariates, relative_change,
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
        all_massif_names += [None]
    else:
        all_massif_names = [None]

    # Illustrate the trend of each massif
    with_significance = False
    for relative_change in [True, False][:]:
        for massif_name in all_massif_names:
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
            for return_period in return_periods_for_plots:
                categories_list = return_period_to_categories_list[return_period]
                for categories in categories_list:
                    categories = set(categories)
                    visualizers_categories = [v for v in visualizers if v.study.altitude in categories]
                    if len(visualizers_categories) > 0:
                        for temperature_covariate in [True, False]:
                            plot_relative_change_at_massif_level(visualizers_categories, massif_name, True,
                                                                 with_significance, relative_change, return_period,
                                                                 snowfall, temperature_covariate)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main()
