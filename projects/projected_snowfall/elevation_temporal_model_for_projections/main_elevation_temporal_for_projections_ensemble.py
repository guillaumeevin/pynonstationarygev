import datetime
import time
from typing import List

import matplotlib as mpl


mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

import matplotlib
from extreme_fit.model.utils import set_seed_for_test

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_snowfall import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from projects.projected_snowfall.elevation_temporal_model_for_projections.ensemble_fit.independent_ensemble_fit import \
    IndependentEnsembleFit
from projects.projected_snowfall.elevation_temporal_model_for_projections.utils_projected_visualizer import \
    load_projected_visualizer_list
from projects.projected_snowfall.elevation_temporal_model_for_projections.visualizer_for_projection_ensemble import \
    VisualizerForProjectionEnsemble
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
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)[:2]
    ensemble_fit_class = [IndependentEnsembleFit]
    temporal_covariate_for_fit = [None, AnomalyTemperatureTemporalCovariate][0]
    set_seed_for_test()
    AbstractExtractEurocodeReturnLevel.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.2

    fast = True
    if fast is None:
        massif_names = None
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        altitudes_list = altitudes_for_groups[2:3]
    elif fast:
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 10
        massif_names = ['Vanoise'][:]
        altitudes_list = altitudes_for_groups[1:2]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    start = time.time()
    main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_classes, ensemble_fit_class, scenario, temporal_covariate_for_fit)
    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


def main_loop(gcm_rcm_couples, altitudes_list, massif_names, study_classes, ensemble_fit_classes, scenario, temporal_covariate_for_fit):
    assert isinstance(altitudes_list, List)
    assert isinstance(altitudes_list[0], List)
    for study_class in study_classes:
        print('Inner loop', study_class)
        visualizer_list = load_projected_visualizer_list(gcm_rcm_couples=gcm_rcm_couples, ensemble_fit_classes=ensemble_fit_classes,
                                                         season=Season.annual, study_class=study_class,
                                                         altitudes_list=altitudes_list, massif_names=massif_names,
                                                         scenario=scenario,
                                                         temporal_covariate_for_fit=temporal_covariate_for_fit)
        for v in visualizer_list:
            v.plot()

        del visualizer_list
        time.sleep(2)





if __name__ == '__main__':
    main()
