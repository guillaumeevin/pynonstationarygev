import numpy as np

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
from extreme_trend.ensemble_simulation.simulation_fit_for_ensemble.abstract_simulation_fit_for_ensemble import \
    AbstractSimulationFitForEnsemble


class SeparateSimulationFitForEnsemble(AbstractSimulationFitForEnsemble):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        for one_fold_fit in one_fold_fits:
            self.print_one_fold_fit_informations(one_fold_fit)
        # Load bootstrap functions
        margin_functions_uncertainty = []
        if AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP > 0:
            for one_fold_fit in one_fold_fits:
                margin_functions_uncertainty.extend(one_fold_fit.bootstrap_fitted_functions_from_fit)
        margin_functions = [one_fold_fit.best_margin_function_from_fit for one_fold_fit in one_fold_fits]
        true_margin_function = self.simulation.simulation_id_to_margin_function[simulation_id]

        return self.compute_dict(margin_functions, margin_functions_uncertainty, true_margin_function)

