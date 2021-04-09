import datetime
import time
from typing import List
import matplotlib

from extreme_fit.model.margin_model.utils import MarginFitMethod
from extreme_trend.ensemble_fit.together_ensemble_fit.together_ensemble_fit import TogetherEnsembleFit
from extreme_trend.one_fold_fit.altitudes_studies_visualizer_for_non_stationary_models import \
    AltitudesStudiesVisualizerForNonStationaryModels
from projects.projected_extreme_snowfall.results.utils import SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE

matplotlib.use('Agg')
import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

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
    rcp_scenarios, rcm_scenarios_extended, get_gcm_list
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    start = time.time()
    study_class = AdamontSnowfall
    ensemble_fit_classes = [IndependentEnsembleFit, TogetherEnsembleFit][:1]
    temporal_covariate_for_fit = [TimeTemporalCovariate,
                                  AnomalyTemperatureWithSplineTemporalCovariate][1]
    set_seed_for_test()
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2
    scenarios = [AdamontScenario.rcp85_extended]

    fast = True
    for scenario in scenarios:
        gcm_rcm_couples = get_gcm_rcm_couples(scenario)
        if fast is None:
            gcm_rcm_couples = gcm_rcm_couples[:2]
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = [1800, 2100]
        elif fast:
            gcm_rcm_couples = gcm_rcm_couples[:2]
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
            altitudes_list = [2400]
        else:
            altitudes_list = [600, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600]

        assert isinstance(gcm_rcm_couples, list)

        altitudes_list = [[a] for a in altitudes_list]
        assert isinstance(altitudes_list, List)
        assert isinstance(altitudes_list[0], List)
        for altitudes in altitudes_list:
            assert len(altitudes) == 1
        AltitudesStudiesVisualizerForNonStationaryModels.consider_at_least_two_altitudes = False

        print('Scenario is', scenario)
        print('Covariate is {}'.format(temporal_covariate_for_fit))

        # Default parameters
        gcm_to_year_min_and_year_max = None
        massif_names = ['Vanoise']
        model_classes = SPLINE_MODELS_FOR_PROJECTION_ONE_ALTITUDE


        visualizer = VisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
            model_classes=model_classes,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            fit_method=MarginFitMethod.evgam,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            remove_physically_implausible_models=True,
            gcm_to_year_min_and_year_max=gcm_to_year_min_and_year_max,
        )
        visualizer.plot()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main()
