import datetime
import time

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_safran import AdamontSnowfall
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario, get_gcm_rcm_couples
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_fit.model.margin_model.polynomial_margin_model.gev_altitudinal_models import StationaryAltitudinal
from extreme_fit.model.margin_model.polynomial_margin_model.models_based_on_pariwise_analysis.gev_with_linear_shape_wrt_altitude import \
    AltitudinalShapeLinearTimeStationary
from extreme_trend.ensemble_fit.independent_ensemble_fit.independent_ensemble_fit import IndependentEnsembleFit
from extreme_trend.ensemble_fit.visualizer_for_projection_ensemble import VisualizerForProjectionEnsemble
from extreme_trend.one_fold_fit.altitude_group import altitudes_for_groups
from projects.projected_swe.weight_computer.abstract_weight_computer import AbstractWeightComputer
from projects.projected_swe.weight_computer.knutti_weight_computer import KnuttiWeightComputer
from projects.projected_swe.weight_computer.non_stationary_weight_computer import NllhWeightComputer
from spatio_temporal_dataset.coordinates.temporal_coordinates.abstract_temporal_covariate_for_fit import \
    TimeTemporalCovariate


def main_weight_computation():
    start = time.time()
    study_class = AdamontSnowfall
    scm_study_class = {
        AdamontSnowfall: SafranSnowfall1Day,
    }[study_class]
    ensemble_fit_classes = [IndependentEnsembleFit]
    temporal_covariate_for_fit = TimeTemporalCovariate
    # model_classes = ALTITUDINAL_GEV_MODELS_BASED_ON_POINTWISE_ANALYSIS
    model_classes = [StationaryAltitudinal, AltitudinalShapeLinearTimeStationary]
    remove_physically_implausible_models = True
    scenario = AdamontScenario.rcp85_extended
    gcm_rcm_couples = get_gcm_rcm_couples(scenario)
    year_min = 1982
    year_max = 2019
    weight_computer_class = [NllhWeightComputer, KnuttiWeightComputer][1]

    fast = True
    if fast is None:
        massif_names = None
        altitudes_list = altitudes_for_groups[2:3]
        gcm_rcm_couples = gcm_rcm_couples[:]
    elif fast:
        massif_names = ['Vanoise', 'Pelvoux', 'Chartreuse'][:1]
        altitudes_list = altitudes_for_groups[1:2]
        gcm_rcm_couples = gcm_rcm_couples[:3]
    else:
        massif_names = None
        altitudes_list = altitudes_for_groups[:]

    visualizer = VisualizerForProjectionEnsemble(
        altitudes_list, gcm_rcm_couples, study_class, Season.annual, scenario,
        model_classes=model_classes,
        ensemble_fit_classes=ensemble_fit_classes,
        massif_names=massif_names,
        temporal_covariate_for_fit=temporal_covariate_for_fit,
        remove_physically_implausible_models=remove_physically_implausible_models,
        gcm_to_year_min_and_year_max={c[0]: (year_min, year_max) for c in gcm_rcm_couples},
    )


    weight_computer = weight_computer_class(visualizer, scm_study_class, year_min, year_max,
                                            sigma_D=10) # type:AbstractWeightComputer
    weight_computer.compute_weights_and_save_them()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)


if __name__ == '__main__':
    main_weight_computation()
    # d = load_gcm_rcm_couple_to_weight(['sd', 'sdf'], [23], 1982, 2019, AdamontScenario.rcp85_extended,
    #                                   weight_class=NllhWeightComputer, gcm_rcm_couple_missing=None)
    # print(d)
