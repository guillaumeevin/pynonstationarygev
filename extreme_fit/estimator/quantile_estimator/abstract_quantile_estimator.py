from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.function.abstract_quantile_function import AbstractQuantileFunction
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.quantile_model.quantile_regression_model import AbstractQuantileRegressionModel
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractQuantileEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset, quantile: float, **kwargs):
        super().__init__(dataset, **kwargs)
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
        linear_margin_function = super().function_from_fit  # type: AbstractMarginFunction
        return AbstractQuantileFunction(linear_margin_function, self.quantile)


class QuantileRegressionEstimator(AbstractQuantileEstimator):

    def __init__(self, dataset: AbstractDataset, quantile: float, quantile_regression_model_class: type, **kwargs):
        super().__init__(dataset, quantile, **kwargs)
        self.quantile_regression_model = quantile_regression_model_class(dataset, quantile) # type: AbstractQuantileRegressionModel

    def _fit(self) -> AbstractResultFromModelFit:
        return self.quantile_regression_model.fit()

    @cached_property
    def quantile_function_from_fit(self) -> AbstractQuantileFunction:
        return self.result_from_model_fit.quantile_function
