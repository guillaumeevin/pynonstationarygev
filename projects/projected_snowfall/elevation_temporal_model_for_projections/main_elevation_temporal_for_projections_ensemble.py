import datetime
import time
from typing import List

import matplotlib as mpl

mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

from extreme_fit.model.margin_model.polynomial_margin_model.utils import \
    ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
from projects.projected_snowfall.elevation_temporal_model_for_projections.visualizer_for_projection_ensemble import \
    MetaVisualizerForProjectionEnsemble
import matplotlib
from extreme_fit.model.utils import set_seed_for_test

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples, \
    rcp_scenarios
from projects.projected_snowfall.elevation_temporal_model_for_projections.ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    AnomalyTemperatureTemporalCovariate

matplotlib.use('Agg')

from projects.altitude_spatial_model.altitudes_fit.plots.plot_histogram_altitude_studies import \
    plot_shoe_plot_changes_against_altitude, plot_histogram_all_trends_against_altitudes

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel

from projects.altitude_spatial_model.altitudes_fit.one_fold_analysis.altitude_group import altitudes_for_groups

from extreme_data.meteo_france_data.scm_models_data.utils import Season


def main():
    study_classes = [AdamontSnowfall][:1]
    scenario = AdamontScenario.rcp85
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    ensemble_fit_class = [IndependentEnsembleFit]
    temporal_covariate_for_fit = [None, AnomalyTemperatureTemporalCovariate][0]
    set_seed_for_test()
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = False
    if fast is None:
        massif_names = None
        gcm_rcm_couples = gcm_rcm_couples[1:2]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = altitudes_for_groups[:2]
    elif fast:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = None
        gcm_rcm_couples = ('CNRM-CM5', 'CCLM4-8-17')
        altitudes_list = altitudes_for_groups[:1]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    start = time.time()
    main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_classes, ensemble_fit_class, scenario,
              temporal_covariate_for_fit)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_classes, ensemble_fit_classes, scenario,
              temporal_covariate_for_fit):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for study_class in study_classes:
        print('Inner loop', study_class)
        model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
        assert scenario in rcp_scenarios

        visualizer = MetaVisualizerForProjectionEnsemble(
            altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
            model_classes=model_classes,
            ensemble_fit_classes=ensemble_fit_classes,
            massif_names=massif_names,
            temporal_covariate_for_fit=temporal_covariate_for_fit,
            confidence_interval_based_on_delta_method=False,
            display_only_model_that_pass_gof_test=False,
            remove_physically_implausible_models=True,
        )
        visualizer.plot()
        del visualizer
        time.sleep(2)


if __name__ == '__main__':
    main()
