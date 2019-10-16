import numpy as np

from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMaxStableEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset, max_stable_model: AbstractMaxStableModel):
        super().__init__(dataset=dataset)
        self.max_stable_model = max_stable_model

    @property
    def max_stable_params_fitted(self):
        raise NotImplementedError

class MaxStableEstimator(AbstractMaxStableEstimator):

    def _fit(self):
        assert self.dataset.maxima_frech(split=self.train_split) is not None
        return self.max_stable_model.fitmaxstab(
            data_frech=self.dataset.maxima_frech_for_spatial_extremes_package(split=self.train_split),
            df_coordinates_spat=self.dataset.df_coordinates(split=self.train_split))

    @property
    def max_stable_params_fitted(self):
        return self.result_from_model_fit.all_parameters
