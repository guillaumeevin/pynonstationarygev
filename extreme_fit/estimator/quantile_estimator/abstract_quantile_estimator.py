from abc import ABC

from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.function.abstract_quantile_function import AbstractQuantileFunction
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractQuantileEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset, quantile: float, **kwargs):
        super().__init__(dataset, **kwargs)
        assert 0 < quantile < 1
        self.quantile = quantile

    @cached_property
    def function_from_fit(self) -> AbstractQuantileFunction:
        raise NotImplementedError


