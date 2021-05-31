

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects


def main_simulation():
    simulation_ids = [0, 1, 2, 3]
    simulation = AbstractSimulationWithEffects(len(simulation_ids))
    plot_simulation(simulation, simulation_ids)


def plot_simulation(simulation, simulation_ids):
    for gev_param_name in [None] + GevParams.PARAM_NAMES:
        simulation.plot_simulation_parameter(gev_param_name, simulation_ids, plot_ensemble_members=True)
    for simulation_id in simulation_ids:
        simulation.plot_time_series(simulation_id=simulation_id)


if __name__ == '__main__':
    main_simulation()
