import time

from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractEstimator(object):
    DURATION = 'Duration'
    MAE_ERROR = 'Mean Average Error'

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset
        self.additional_information = dict()

    def fit(self):
        ts = time.time()
        self._fit()
        te = time.time()
        self.additional_information[self.DURATION] = int((te - ts) * 1000)

    def scalars(self, true_max_stable_params: dict):
        error = self._error(true_max_stable_params)
        return {**error, **self.additional_information}

    # Methods to override in the child class

    def _fit(self):
        pass

    def _error(self, true_max_stable_params: dict):
        pass
