import datetime
import time

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel, NonStationaryLocationAndScaleTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions import SimulationVersion1, \
    SimulationVersion2, SimulationVersion3
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v2 import \
    SimulationLogScaleWithShift, SimulationLogScaleWithoutShift
from extreme_trend.ensemble_simulation.visualizer_for_simulation_ensemble import VisualizerForSimulationEnsemble
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projects.projected_extreme_snowfall.results.setting_utils import LINEAR_MODELS_FOR_PROJECTION_ONE_ALTITUDE


def main_simulation():
    start = time.time()

    model_classes = [NonStationaryLocationAndScaleAndShapeTemporalModel]
    # OneFoldFit.SELECTION_METHOD_NAME = 'aic'
    OneFoldFit.SELECTION_METHOD_NAME = 'split_sample'
    fast = None
    if fast is True:
        nb_simulations = 1
        year_list_to_test = [2025, 2050, 2075, 2100]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0
    elif fast is None:
        nb_simulations = 4
        year_list_to_test = [2020 + i * 5 for i in range(17)]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0
    else:
        nb_simulations = 100
        year_list_to_test = list(range(2020, 2101))
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 5

    # Set settings
    simulation_version = SimulationLogScaleWithShift
    # simulation_version = [SimulationVersion1, SimulationVersion2, SimulationVersion3,
    #                       SimulationLogScaleWithShift, SimulationLogScaleWithoutShift][-1]
    simulation = simulation_version(nb_simulations)
    visualizer = VisualizerForSimulationEnsemble(simulation, year_list_to_test,
                                                 return_period=50,
                                                 model_classes=model_classes,
                                                 fast=fast)
    visualizer.plot_mean_metrics()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)

if __name__ == '__main__':
    main_simulation()
