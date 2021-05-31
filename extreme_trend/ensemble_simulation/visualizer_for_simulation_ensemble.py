from typing import List

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.together_simulation_fit_for_ensemble import \
    TogetherSimulationFitForEnsemble
import matplotlib.pyplot as plt


class VisualizerForSimulationEnsemble(StudyVisualizer):

    def __init__(self, simulation, year_list_to_test, return_period, model_classes):
        super().__init__(SafranSnowfall1Day(), show=False, save_to_file=True)
        # Load simulation fits
        self.simulation_fits = []  # type: List[AbstractSimulationFitForEnsemble]
        for with_effects in [True, False][:]:
            self.simulation_fits.append(TogetherSimulationFitForEnsemble(simulation, year_list_to_test, return_period, model_classes,
                                                                         with_effects=with_effects, with_observation=True))
        self.return_period = return_period

    def plot_mean_metrics(self):
        for metric_name in AbstractSimulationFitForEnsemble.METRICS[:1]:
            self.plot_mean_metric(metric_name)

    def plot_mean_metric(self, metric_name):
        ax = plt.gca()
        for simulation_fit in self.simulation_fits:
            simulation_fit.plot_mean_metric(ax, metric_name)
        ax.legend()
        self.show_or_save_to_file(plot_name='mean {} for {}-year return level'.format(metric_name, self.return_period))
        plt.close()
