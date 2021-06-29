from typing import List
import os.path as op

import pandas as pd

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.visualization.study_visualizer import StudyVisualizer
from extreme_data.utils import DATA_PATH
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
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit
from projects.projected_extreme_snowfall.results.combination_utils import number_to_sub_numbers
from projects.projected_extreme_snowfall.results.part_2.v2.utils import load_excel, add_dynamical_value
from root_utils import get_display_name_from_object_type

SIMULATION_PATH = op.join(DATA_PATH, "simulation_excel")


class VisualizerForSimulationEnsemble(StudyVisualizer):

    def __init__(self, simulation: AbstractSimulationWithEffects, year_list_to_test, return_period, model_classes, fast):
        super().__init__(SafranSnowfall1Day(), show=False, save_to_file=True)
        self.model_classes = model_classes
        self.simulation = simulation
        self.return_period = return_period
        # Load simulation fits
        self.simulation_fits = []  # type: List[AbstractSimulationFitForEnsemble]
        fit_classes = [SeparateSimulationFitForEnsemble, TogetherSimulationFitForEnsemble][:]
        fit_class_to_colors = {
            SeparateSimulationFitForEnsemble: ['silver', 'gray', 'black'],
            TogetherSimulationFitForEnsemble: ['tan', 'peru', 'brown']
        }
        for fit_class in fit_classes:
            colors = fit_class_to_colors[fit_class]
            fit_class_simulation_fits = [
                fit_class(self.simulation, year_list_to_test, return_period, model_classes,
                          with_effects=False, with_observation=False, color=colors[0]),
                fit_class(self.simulation, year_list_to_test, return_period, model_classes,
                          with_effects=True, with_observation=True, color=colors[2]),

            ][:]
            self.simulation_fits.extend(fit_class_simulation_fits)
        self.simulation_fits = self.simulation_fits[::-1][:]
        if fast:
            self.simulation_fits = self.simulation_fits[1:2]



    def write_to_csv(self, ):
        class_name = get_display_name_from_object_type(type(self.simulation))
        if  '__' not in class_name:
            print(class_name, 'not a valid name for csv writing')
        else:
            for metric_name in AbstractSimulationFitForEnsemble.METRICS[:]:

                class_name_prefix, row_name, column_name = class_name.split('__')
                excel_filepath = op.join(SIMULATION_PATH, "{}_{}_{}_{}.xlsx".format(metric_name, class_name_prefix,
                                                                                    self.simulation.nb_simulations,
                                                                                    AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP))

                for j, simulation_fit in enumerate(self.simulation_fits, 1):
                    print('\n\nSimulation fit method #{}'.format(j))
                    simulation_fit.update_csv(excel_filepath, metric_name, row_name, column_name)

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
        plot_name = ylabel
        sub_combination = '_'.join([str(n) for n in number_to_sub_numbers[2]])
        plot_name += '_' + OneFoldFit.SELECTION_METHOD_NAME
        plot_name += '_models{}_combination_{}'.format(len(self.model_classes), sub_combination)
        plot_name += '_bootstrap{}_nbsimu{}'.format(AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP,
                                                           self.simulation.nb_simulations)  + self.simulation.summary_parameter
        self.show_or_save_to_file(plot_name=plot_name)
        plt.close()
