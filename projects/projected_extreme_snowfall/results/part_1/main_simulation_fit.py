

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.utils import set_seed_for_test
from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.visualizer_for_simulation_ensemble import VisualizerForSimulationEnsemble
from projects.projected_extreme_snowfall.results.setting_utils import LINEAR_MODELS_FOR_PROJECTION_ONE_ALTITUDE


def main_simulation():
    model_classes = LINEAR_MODELS_FOR_PROJECTION_ONE_ALTITUDE

    fast = True
    if fast:
        model_classes = model_classes[:3]
        nb_simulations = 1
        year_list_to_test = [2025, 2050, 2075, 2100]
    else:
        nb_simulations = 1
        year_list_to_test = list(range(2020, 2101))

    # Set settings
    simulation = AbstractSimulationWithEffects(nb_simulations)
    visualizer = VisualizerForSimulationEnsemble(simulation, year_list_to_test,
                                                 return_period=50,
                                                 model_classes=model_classes)
    visualizer.plot_mean_metrics()




if __name__ == '__main__':
    main_simulation()
