import numpy as np

from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    AbstractExtractEurocodeReturnLevel
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
        # Load one fold fit
        one_fold_fit = self.load_one_fold_fit(dataset, str(simulation_id))
        nb_bootstrap = AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP
        if AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP == 0:
            margin_functions_uncertainty = []
        else:
            OneFoldFit.multiprocessing = False
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP *= self.simulation.nb_ensemble_member
            margin_functions_uncertainty = one_fold_fit.bootstrap_fitted_functions_from_fit
            AbstractExtractEurocodeReturnLevel.NB_BOOTSTRAP = nb_bootstrap
        margin_function = one_fold_fit.best_margin_function_from_fit
        true_margin_function = self.simulation.simulation_id_to_margin_function[simulation_id]

        return self.compute_dict([margin_function], margin_functions_uncertainty, true_margin_function)
