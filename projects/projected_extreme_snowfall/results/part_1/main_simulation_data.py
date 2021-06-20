

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions import SimulationVersion2
from extreme_trend.ensemble_simulation.simulation_generator_with_effect.simulation_versions_v2 import \
    SimulationLogScaleWithShift10


def main_simulation():
    # nb = 1
    nb = 4
    simulation_ids = list(range(nb))
    simulation_class = [SimulationLogScaleWithShift10][0]
    simulation = simulation_class(len(simulation_ids))
    plot_simulation(simulation, simulation_ids)


def plot_simulation(simulation, simulation_ids):
    for simulation_id in simulation_ids:
        simulation.plot_bias(simulation_id)
        simulation.plot_time_series(simulation_id=simulation_id)
    for gev_param_name in GevParams.PARAM_NAMES + [None]:
        simulation.plot_simulation_parameter(gev_param_name, simulation_ids, plot_ensemble_members=True)


if __name__ == '__main__':
    main_simulation()
