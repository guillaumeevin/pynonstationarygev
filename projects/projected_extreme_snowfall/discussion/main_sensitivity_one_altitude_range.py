import datetime
import time
from typing import List
import matplotlib


matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_trend.ensemble_fit.abstract_ensemble_fit import AbstractEnsembleFit
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit

from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.ensemble_fit.visualizer_for_sensitivity import VisualizerForSensivity
from spatio_temporal_dataset.coordinates.temporal_coordinates.temperature_covariate import \
    AnomalyTemperatureWithSplineTemporalCovariate

from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS

from extreme_fit.model.utils import set_seed_for_test

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples, \
    rcp_scenarios, rcm_scenarios_extended
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()
    study_class = AdamontSnowfall
    ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][1:]
    set_seed_for_test()
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = False
    scenarios = [AdamontScenario.rcp85]
    scenarios = rcp_scenarios[1:]
    scenarios = rcm_scenarios_extended[2:][::-1]

    if fast in [None, True]:
        scenarios = scenarios[:1]

    for scenario in scenarios:
        gcm_rcm_couples = get_gcm_rcm_couples(scenario)
        if fast is None:
            massif_names = None
            gcm_rcm_couples = gcm_rcm_couples[:2]
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = altitudes_for_groups[:1]
        elif fast:
            massif_names = ['Vanoise', 'Haute-Maurienne', 'Vercors', 'Ubaye']
            gcm_rcm_couples = gcm_rcm_couples[4:6]
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = altitudes_for_groups[1:2]
        else:
            massif_names = None
            altitudes_list = altitudes_for_groups[2:3]

        assert isinstance(gcm_rcm_couples, list)

        assert isinstance(altitudes_list, List)
        assert isinstance(altitudes_list[0], List)
        print('Scenario is', scenario)

        model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
        assert scenario is not AdamontScenario.histo
        remove_physically_implausible_models = True
        temp_cov = True
        temporal_covariate_for_fit = AnomalyTemperatureWithSplineTemporalCovariate if temp_cov else TimeTemporalCovariate
        print('Covariate is {}'.format(temporal_covariate_for_fit))

        for altitudes in altitudes_list:
            sub_altitudes_list = [altitudes]
            print(sub_altitudes_list)
            for is_temperature_interval in [True, False][:1]:
                for is_shift_interval in [True, False][1:]:
                    visualizer = VisualizerForSensivity(
                        sub_altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
                        model_classes=model_classes,
                        ensemble_fit_classes=ensemble_fit_classes,
                        massif_names=massif_names,
                        temporal_covariate_for_fit=temporal_covariate_for_fit,
                        remove_physically_implausible_models=remove_physically_implausible_models,
                        is_temperature_interval=is_temperature_interval,
                        is_shift_interval=is_shift_interval,
                    )
                    visualizer.plot()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main()
