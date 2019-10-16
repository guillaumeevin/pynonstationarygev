from typing import Union

from cached_property import cached_property

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractEstimator(object):

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset  # type: AbstractDataset
        self._result_from_fit = None  # type: Union[None, AbstractResultFromModelFit]

    # Class constructor
    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        raise NotImplementedError

    # Fit estimator

    def fit(self):
        self._result_from_fit = self._fit()

    def _fit(self) -> AbstractResultFromModelFit:
        raise NotImplementedError

    # Results from model fit

    @property
    def result_from_model_fit(self) -> AbstractResultFromModelFit:
        assert self._result_from_fit is not None, 'Estimator has not been fitted'
        return self._result_from_fit

    @cached_property
    def margin_function_from_fit(self) -> AbstractMarginFunction:
        raise NotImplementedError

    # Short cut properties

    @property
    def train_split(self):
        return self.dataset.train_split




