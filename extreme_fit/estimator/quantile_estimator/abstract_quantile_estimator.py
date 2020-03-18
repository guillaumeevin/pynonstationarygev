from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.quantile_estimator.abstract_quantile_function import AbstractQuantileFunction
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractQuantileEstimator(AbstractEstimator):

    def __init__(self, quantile: float, **kwargs):
        super().__init__(**kwargs)
        assert 0 < quantile < 1
        self.quantile = quantile

    @cached_property
    def quantile_function_from_fit(self) -> AbstractQuantileFunction:
        pass


class QuantileEstimatorFromMargin(AbstractQuantileEstimator, LinearMarginEstimator):

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel, quantile):
        super().__init__(dataset=dataset, quantile=quantile, margin_model=margin_model)

    @cached_property
    def quantile_function_from_fit(self) -> AbstractQuantileFunction:
        linear_margin_function = super().margin_function_from_fit
        return AbstractQuantileFunction(linear_margin_function, self.quantile)
