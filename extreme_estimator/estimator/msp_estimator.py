from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.R_fit.max_stable_fit.max_stable_models import MaxStableModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
import numpy as np


class MaxStableEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset, max_stable_model: MaxStableModel):
        self.dataset = dataset
        self.max_stable_model = max_stable_model
        self.max_stable_params_fitted = None

    def fit(self):
        self.max_stable_params_fitted = self.max_stable_model.fitmaxstab(maxima=self.dataset.maxima, coord=self.dataset.coord)

    def error(self, true_max_stable_params: dict):
        absolute_errors = {param_name: np.abs(param_true_value - self.max_stable_params_fitted[param_name])
                           for param_name, param_true_value in true_max_stable_params.items()}
        mean_absolute_error = np.mean(np.array(list(absolute_errors.values())))
        # return {**absolute_errors, **{'mae': mean_absolute_error}}
        return mean_absolute_error
