import datetime
import time

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.model.margin_model.spline_margin_model.temporal_spline_model_degree_1 import \
    NonStationaryTwoLinearLocationAndScaleAndShapeModel
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
from projects.projected_extreme_snowfall.results.get_nb_linear_pieces import run_selection
from projects.projected_extreme_snowfall.results.setting_utils import set_up_and_load

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()

    fast = False
    snowfall = None
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method, season = set_up_and_load(
        fast, snowfall)

    # altitudes_list = [[900]]
    # altitudes_list = [[1500]]

    altitudes_list = [[2100]]
    altitudes_list = [[2700]]
    altitudes_list = [[3300]]
    altitudes_list = [[2100], [2400], [2700], [3000], [3300], [3600]][:]

    print('altitude', altitudes_list)

    ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]
    massif_names = AbstractStudy.all_massif_names()[:]
    # massif_names = ['Mercantour', 'Thabor', 'Devoluy', 'Parpaillon', 'Haut_Var-Haut_Verdon'][:2]

    for altitudes in altitudes_list:
        massif_names, massif_name_to_model_class, massif_name_to_parametrization_number, linear_effects = run_selection(massif_names,
                                                                                          altitudes[0],
                                                                                                        gcm_rcm_couples,
                                                                                                        safran_study_class,
                                                                                                        scenario,
                                                                                                        study_class,
                                                                                          snowfall=snowfall,
                                                                                                                        season=season)

        massif_name_to_param_name_to_climate_coordinates_with_effects = {}
        for massif_name, parametrization_number in massif_name_to_parametrization_number.items():
            combination = (parametrization_number, parametrization_number, 0)
            param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(combination)
            massif_name_to_param_name_to_climate_coordinates_with_effects[massif_name] = param_name_to_climate_coordinates_with_effects

        visualizer = VisualizerForProjectionEnsemble(
            [altitudes], gcm_rcm_couples, study_class, season, scenario,
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

        with_significance = False
        sub_visualizers = [together_ensemble_fit.visualizer
                           for together_ensemble_fit in visualizer.ensemble_fits(TogetherEnsembleFit)]
        print(len(sub_visualizers))
        sub_visualizer = sub_visualizers[0]
        sub_visualizer.plot_moments_projections(with_significance, scenario)

        end = time.time()
        duration = str(datetime.timedelta(seconds=end - start))
        print('Total duration', duration)


if __name__ == '__main__':
    main()
