from collections import OrderedDict

import numpy as np

from extreme_trend.ensemble_simulation.abstract_simulation_with_effect import AbstractSimulationWithEffects
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble
from extreme_trend.one_fold_fit.one_fold_fit import OneFoldFit


class TogetherSimulationFitForEnsemble(AbstractSimulationFitForEnsemble):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.with_observation:
            raise NotImplementedError('create a dataset without obs')
        self.simulation_id_to_one_fold_fit = OrderedDict()

    @property
    def name(self):
        name = "Together fit "
        if self.with_effects:
            name += 'with effects'
        else:
            name += 'without effects'
        return name

    def compute_metric_name_to_list(self, simulation_id):
        dataset = self.simulation.simulation_id_to_dataset[simulation_id]
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
        for x in self.x_list_to_test:
            coordinates = np.array([x])
            true_value = self.compute_return_levels(true_margin_function, coordinates)
            # Compute rmse
            prediction = self.compute_return_levels(margin_function, coordinates)
            rmse = np.power(true_value - prediction, 2)
            rmse_list.append(rmse)
            # Compute crpss from bootstrap samples
            # predictions = [self.compute_return_levels(f, x) for f in margin_functions]
        return {
            self.RMSE_METRIC: rmse_list,
            self.CRPSS_METRIC: crpss_list
        }


