from typing import List

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.separate_simulation_fit_for_ensemble import \
    SeparateSimulationFitForEnsemble
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.together_simulation_fit_for_ensemble import \
    TogetherSimulationFitForEnsemble
import matplotlib.pyplot as plt

from extreme_trend.ensemble_simulation.simulation_generator_with_effect.abstract_simulation_with_effect import \
    AbstractSimulationWithEffects


class VisualizerForSimulationEnsemble(StudyVisualizer):

    def __init__(self, simulation: AbstractSimulationWithEffects, year_list_to_test, return_period, model_classes, fast):
        super().__init__(SafranSnowfall1Day(), show=False, save_to_file=True)
        self.simulation = simulation
        self.return_period = return_period
        # Load simulation fits
        self.simulation_fits = []  # type: List[AbstractSimulationFitForEnsemble]
        fit_classes = [SeparateSimulationFitForEnsemble, TogetherSimulationFitForEnsemble][:]
        fit_class_to_colors = {
            SeparateSimulationFitForEnsemble: ['silver', 'gray'],
            TogetherSimulationFitForEnsemble: ['tan', 'brown']
        }
        for fit_class in fit_classes:
            colors = fit_class_to_colors[fit_class]
            fit_class_simulation_fits = [
                fit_class(self.simulation, year_list_to_test, return_period, model_classes,
                          with_effects=False, with_observation=False, color=colors[0]),
                # fit_class(simulation, year_list_to_test, return_period, model_classes,
                #           with_effects=False, with_observation=True, color='red'),
                fit_class(self.simulation, year_list_to_test, return_period, model_classes,
                          with_effects=True, with_observation=True, color=colors[1])
            ][:]
            self.simulation_fits.extend(fit_class_simulation_fits)
        self.simulation_fits = self.simulation_fits[::-1][:]

    def plot_mean_metrics(self):
        for metric_name in AbstractSimulationFitForEnsemble.METRICS[:]:
            self.plot_mean_metric(metric_name)

    def plot_mean_metric(self, metric_name):
        ax = plt.gca()
        for j, simulation_fit in enumerate(self.simulation_fits, 1):
            print('\n\nSimulation fit method #{}'.format(j))
            simulation_fit.plot_mean_metric(ax, metric_name)
        ax.legend(ncol=2, prop={'size': 6})
        ylabel = 'Mean {} for {}-year return level'.format(metric_name.capitalize(), self.return_period)
        ax.set_ylabel(ylabel)
        plot_name = ylabel + 'bootstrap{}_nbsimu{}'.format(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP,
                                                           self.simulation.nb_simulations)  + self.simulation.summary_parameter
        self.show_or_save_to_file(plot_name=plot_name)
        plt.close()
