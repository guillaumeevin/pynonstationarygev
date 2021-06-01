from typing import List

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.separate_simulation_fit_for_ensemble import \
    SeparateSimulationFitForEnsemble
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.together_simulation_fit_for_ensemble import \
    TogetherSimulationFitForEnsemble
import matplotlib.pyplot as plt


class VisualizerForSimulationEnsemble(StudyVisualizer):

    def __init__(self, simulation, year_list_to_test, return_period, model_classes):
        super().__init__(SafranSnowfall1Day(), show=False, save_to_file=True)
        self.return_period = return_period
        # Load simulation fits
        self.simulation_fits = []  # type: List[AbstractSimulationFitForEnsemble]
        fit_classes = [SeparateSimulationFitForEnsemble, TogetherSimulationFitForEnsemble][:]
        for fit_class in fit_classes:
            fit_class_simulation_fits = [
                fit_class(simulation, year_list_to_test, return_period, model_classes,
                          with_effects=False, with_observation=False, color='blue'),
                fit_class(simulation, year_list_to_test, return_period, model_classes,
                          with_effects=False, with_observation=True, color='red'),
                fit_class(simulation, year_list_to_test, return_period, model_classes,
                          with_effects=True, with_observation=True, color="green")
            ][:]
            self.simulation_fits.extend(fit_class_simulation_fits)

    def plot_mean_metrics(self):
        for metric_name in AbstractSimulationFitForEnsemble.METRICS[:1]:
            self.plot_mean_metric(metric_name)

    def plot_mean_metric(self, metric_name):
        ax = plt.gca()
        for simulation_fit in self.simulation_fits:
            simulation_fit.plot_mean_metric(ax, metric_name)
        ax.legend(ncol=2, prop={'size': 5})
        ylabel = 'Mean {} for {}-year return level'.format(metric_name.capitalize(), self.return_period)
        ax.set_ylabel(ylabel)
        self.show_or_save_to_file(plot_name=ylabel)
        plt.close()
