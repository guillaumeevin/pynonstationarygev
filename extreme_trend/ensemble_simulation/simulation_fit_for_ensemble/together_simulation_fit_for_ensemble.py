from collections import OrderedDict

import numpy as np

from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class TogetherSimulationFitForEnsemble(AbstractSimulationFitForEnsemble):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def name(self):
        return self.add_suffix("Together fit ")

    def compute_metric_name_to_list(self, simulation_id):
        if self.with_observation:
            dataset = self.simulation.simulation_id_to_together_dataset_with_obs[simulation_id]
        else:
            dataset = self.simulation.simulation_id_to_together_dataset_without_obs[simulation_id]
        one_fold_fit = OneFoldFit(massif_name=str(simulation_id), dataset=dataset, models_classes=self.model_classes,
                                  altitude_group=self.altitude_group,
                                  only_models_that_pass_goodness_of_fit_test=self.only_models_that_pass_goodness_of_fit_test,
                                  confidence_interval_based_on_delta_method=self.confidence_interval_based_on_delta_method,
                                  param_name_to_climate_coordinates_with_effects=self.param_name_to_climate_coordinates_with_effects,
                                  fit_method=self.fit_method)
        # todo: it should be the bootstrap here
        margin_functions = []
        margin_function = one_fold_fit.best_margin_function_from_fit
        true_margin_function = self.simulation.simulation_id_to_margin_function[simulation_id]
        crpss_list = []
        rmse_list = []
        absolute_list = []
        for x in self.x_list_to_test:
            coordinates = np.array([x])
            true_value = self.compute_return_levels(true_margin_function, coordinates)
            prediction = self.compute_return_levels(margin_function, coordinates)
            # Compute rmse
            rmse = np.power(true_value - prediction, 2)
            rmse_list.append(rmse)
            # Compute absolute relative difference
            absolute_relative_difference = np.abs( 100 * (true_value - prediction) / true_value)
            absolute_list.append(absolute_relative_difference)


            # Compute crpss from bootstrap samples
            # predictions = [self.compute_return_levels(f, x) for f in margin_functions]
        return {
            self.RMSE_METRIC: rmse_list,
            self.CRPSS_METRIC: crpss_list,
            self.ABSOLUTE_RELATIVE_DIFFERENCE_METRIC: absolute_list,
        }
