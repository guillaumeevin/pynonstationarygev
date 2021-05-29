from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.temporal_linear_margin_models import \
    NonStationaryLocationAndScaleAndShapeTemporalModel
from extreme_fit.model.utils import set_seed_for_test
from projects.projected_extreme_snowfall.results.part_1.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects


def main_simulation():
    set_seed_for_test()
    simulation_ids = [0, 1, 2, 3]
    simulation = AbstractSimulationWithEffects(len(simulation_ids))
    for gev_param_name in [None] + GevParams.PARAM_NAMES:
        simulation.plot_simulation_parameter(gev_param_name, simulation_ids, plot_ensemble_members=True)
    for simulation_id in simulation_ids:
        simulation.plot_time_series(simulation_id=simulation_id)


if __name__ == '__main__':
    main_simulation()
