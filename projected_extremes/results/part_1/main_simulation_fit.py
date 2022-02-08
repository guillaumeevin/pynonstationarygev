import datetime
import time

from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_version_for_excel import \
    ShiftExperiment__0__0, ShiftExperiment__0__10, ShiftExperiment__10__0, ShiftExperiment__20__20, \
    ShiftExperiment__10__20, ShiftExperiment__20__10
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_version_v4 import \
    CenterExperiment7_5, CenterExperiment10, CenterExperiment12_5
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_version_v5 import MeanExperiment7_5, \
    MeanExperiment5, MeanExperiment10, MeanExperiment12_5
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_version_v6 import STDExperiment5, \
    STDExperiment7_5, STDExperiment10, STDExperiment12_5
from extreme_trend.ensemble_simulation.visualizer_for_simulation_ensemble import VisualizerForSimulationEnsemble
from projected_extremes.results.part_1.main_simulation_data import plot_simulation


def main_simulation():
    start = time.time()

    model_classes = [NonStationaryLocationAndScaleAndShapeTemporalModel]

    fast = True
    if fast is True:
        nb_simulations = 1
        year_list_to_test = [2025, 2050, 2075, 2100]
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0
    elif fast is None:
        nb_simulations = 10
        year_list_to_test = [2020 + i * 5 for i in range(17)]
        # AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0
    else:
        nb_simulations = 100
        year_list_to_test = list(range(2020, 2101))
        AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = 0

    # Set settings
    simulation_class = [CenterExperiment7_5, CenterExperiment10, CenterExperiment12_5][0]
    simulation_class = [CenterExperiment7_5, CenterExperiment10, CenterExperiment12_5][0]
    simulation_class = [MeanExperiment5, MeanExperiment7_5, MeanExperiment10, MeanExperiment12_5][3]
    simulation_class = [STDExperiment5, STDExperiment7_5, STDExperiment10, STDExperiment12_5][0]
    simulation_classes = [ShiftExperiment__0__0, ShiftExperiment__10__0, ShiftExperiment__0__10]
    simulation_classes = [ShiftExperiment__10__20, ShiftExperiment__20__10, ShiftExperiment__20__20]
    # simulation_classes = [ShiftExperiment__10__10, ShiftExperiment__0__20, ShiftExperiment__20__0]


    # simulation_class = [SimulationSnowLoadWithShiftLikeSafran, SimulationSnowLoadWithShift0And0,
    #                     SimulationLogScaleWithShift10And0, SimulationLogScaleWithShift0And10][0]
    # simulation_version = [SimulationVersion1, SimulationVersion2, SimulationVersion3,
    #                       SimulationLogScaleWithShift, SimulationLogScaleWithoutShift][-1]
    for simulation_class in simulation_classes:
        simulation = simulation_class(nb_simulations)
        plot_simulation(simulation, list(range(nb_simulations)))
        visualizer = VisualizerForSimulationEnsemble(simulation, year_list_to_test,
                                                     return_period=50,
                                                     model_classes=model_classes,
                                                     fast=fast)
        visualizer.write_to_csv()
    # visualizer.plot_mean_metrics()

    end = time.time()
    duration = str(datetime.timedelta(seconds=end - start))
    print('Total duration', duration)

if __name__ == '__main__':
    main_simulation()
