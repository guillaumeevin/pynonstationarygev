import datetime
import time

import matplotlib

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from projects.projected_extreme_snowfall.results.combination_utils import \
    load_param_name_to_climate_coordinates_with_effects
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

    fast = None
    snowfall = False
    altitudes_list, gcm_rcm_couples, massif_names, model_classes, scenario, \
    study_class, temporal_covariate_for_fit, remove_physically_implausible_models, \
    display_only_model_that_pass_gof_test, safran_study_class, fit_method = set_up_and_load(
        fast, snowfall)

    ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]

    combination = (1, 1, 0)
    param_name_to_climate_coordinates_with_effects = load_param_name_to_climate_coordinates_with_effects(combination)

    all_massif_names = AbstractStudy.all_massif_names()[:]
    # AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 100
    all_massif_names = ['Vanoise']
    for massif_name in all_massif_names:
        massif_names = [massif_name]
        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
            model_classes=model_classes,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            fit_method=fit_method,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            safran_study_class=safran_study_class,
            display_only_model_that_pass_gof_test=display_only_model_that_pass_gof_test,
            param_name_to_climate_coordinates_with_effects=param_name_to_climate_coordinates_with_effects,
        )
        visualizer.plot()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main()
