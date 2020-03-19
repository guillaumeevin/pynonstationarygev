from abc import ABC

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.function.abstract_quantile_function import AbstractQuantileFunction, \
    QuantileFunctionFromMarginFunction, QuantileFunctionFromParamFunction
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.param_function import LinearParamFunction
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.quantile_model.quantile_regression_model import AbstractQuantileRegressionModel
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_quantilreg import ResultFromQuantreg
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractQuantileEstimator(AbstractEstimator, ABC):

    def __init__(self, dataset: AbstractDataset, quantile: float, **kwargs):
        super().__init__(dataset, **kwargs)
        assert 0 < quantile < 1
        self.quantile = quantile

    @cached_property
    def function_from_fit(self) -> AbstractQuantileFunction:
        raise NotImplementedError


class QuantileEstimatorFromMargin(LinearMarginEstimator, AbstractQuantileEstimator):

    def __init__(self, dataset: AbstractDataset, margin_model: AbstractTemporalLinearMarginModel, quantile):
        super().__init__(dataset=dataset, quantile=quantile, margin_model=margin_model)

    @cached_property
    def function_from_fit(self) -> AbstractQuantileFunction:
        linear_margin_function = super().function_from_fit  # type: AbstractMarginFunction
        return QuantileFunctionFromMarginFunction(self.dataset.coordinates, linear_margin_function, self.quantile)


class QuantileRegressionEstimator(AbstractQuantileEstimator):

    def __init__(self, dataset: AbstractDataset, quantile: float, quantile_regression_model_class: type, **kwargs):
        super().__init__(dataset, quantile, **kwargs)
        self.quantile_regression_model = quantile_regression_model_class(dataset, quantile)  # type: AbstractQuantileRegressionModel

    def _fit(self) -> AbstractResultFromModelFit:
        return self.quantile_regression_model.fit()

    @cached_property
    def function_from_fit(self) -> AbstractQuantileFunction:
        result_from_model_fit = self.result_from_model_fit  # type: ResultFromQuantreg
        coefs = result_from_model_fit.coefficients
        nb_coefs = len(coefs)
        dims = list(range(nb_coefs-1))
        idx_to_coef = dict(zip([-1] + dims, coefs))
        linear_coef = LinearCoef(idx_to_coef=idx_to_coef)
        param_function = LinearParamFunction(dims=dims, coordinates=self.dataset.coordinates.coordinates_values(),
                                             linear_coef=linear_coef)
        return QuantileFunctionFromParamFunction(coordinates=self.dataset.coordinates,
                                                 param_function=param_function)
