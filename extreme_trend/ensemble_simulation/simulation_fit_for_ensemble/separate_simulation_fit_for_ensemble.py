from collections import OrderedDict

import numpy as np

from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class SeparateSimulationFitForEnsemble(AbstractSimulationFitForEnsemble):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def linestyle(self):
        return 'dotted'

    @property
    def name(self):
        return self.add_suffix("Separate fit ")

    def compute_metric_name_to_list(self, simulation_id):
        if self.with_observation:
            datasets = self.simulation.simulation_id_to_separate_datasets_with_obs[simulation_id]
        else:
            datasets = self.simulation.simulation_id_to_separate_datasets_without_obs[simulation_id]
        # Fit one fold
        one_fold_fits = [self.load_one_fold_fit(dataset, "name") for dataset in datasets]
        # todo: it should be the bootstrap here
        margin_functions_uncertainty = []
        margin_functions = [one_fold_fit.best_margin_function_from_fit for one_fold_fit in one_fold_fits]
        true_margin_function = self.simulation.simulation_id_to_margin_function[simulation_id]
        crpss_list = []
        rmse_list = []
        absolute_list = []
        for x in self.x_list_to_test:
            coordinates = np.array([x])
            prediction = np.mean([self.compute_return_levels(f, coordinates) for f in margin_functions])
            predictions_with_uncertainty = [self.compute_return_levels(f, x) for f in margin_functions_uncertainty]

            absolute_relative_difference, crpss, rmse = self.compute_metrics(coordinates, prediction, predictions_with_uncertainty,
                                                                             true_margin_function)

            rmse_list.append(rmse)
            absolute_list.append(absolute_relative_difference)
            crpss_list.append(crpss)

        return {
            self.RMSE_METRIC: rmse_list,
            self.CRPSS_METRIC: crpss_list,
            self.ABSOLUTE_RELATIVE_DIFFERENCE_METRIC: absolute_list,
        }

