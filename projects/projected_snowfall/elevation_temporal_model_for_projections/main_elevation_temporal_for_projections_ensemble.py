import datetime
import time
from typing import List

import matplotlib as mpl

from projects.projected_snowfall.elevation_temporal_model_for_projections.visualizer_for_sensitivity import \
    VisualizerForSensivity

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.projected_snowfall.elevation_temporal_model_for_projections.visualizer_for_projection_ensemble import \
    VisualizerForProjectionEnsemble
import matplotlib
from extreme_fit.model.utils import set_seed_for_test

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples, \
    rcp_scenarios
from projects.projected_snowfall.elevation_temporal_model_for_projections.independent_ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    AnomalyTemperatureTemporalCovariate, TimeTemporalCovariate

matplotlib.use('Agg')

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()
    study_class = AdamontSnowfall
    ensemble_fit_class = [IndependentEnsembleFit]
    temporal_covariate_for_fit = [TimeTemporalCovariate, AnomalyTemperatureTemporalCovariate][1]
    set_seed_for_test()
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = None
    sensitivity_plot = True
    scenarios = rcp_scenarios if fast is False else [AdamontScenario.rcp85]

    for scenario in scenarios:
        gcm_rcm_couples = get_gcm_rcm_couples(scenario)
        if fast is None:
            massif_names = None
            gcm_rcm_couples = None
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = altitudes_for_groups[3:]
        elif fast:
            massif_names = ['Vanoise', 'Haute-Maurienne']
            gcm_rcm_couples = gcm_rcm_couples[4:6]
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = altitudes_for_groups[2:]
        else:
            massif_names = None
            altitudes_list = altitudes_for_groups[:]

        assert isinstance(gcm_rcm_couples, list)

        main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_class, ensemble_fit_class, scenario,
                  temporal_covariate_for_fit, sensitivity_plot=sensitivity_plot)

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_class, ensemble_fit_classes, scenario,
              temporal_covariate_for_fit, sensitivity_plot=False):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    print('Covariate is {}'.format(temporal_covariate_for_fit))

    model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    assert scenario in rcp_scenarios
    remove_physically_implausible_models = True

    if sensitivity_plot:
        visualizer = VisualizerForSensivity(
            altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
            model_classes=model_classes,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
        )
    else:
        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
            model_classes=model_classes,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=remove_physically_implausible_models,
            gcm_to_year_min_and_year_max=None,
        )
    visualizer.plot()


if __name__ == '__main__':
    main()
