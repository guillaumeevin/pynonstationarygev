from abc import ABC

import numpy as np
from cached_property import cached_property

from extreme_fit.estimator.abstract_estimator import AbstractEstimator
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.quantile_estimator.abstract_quantile_estimator import AbstractQuantileEstimator
from extreme_fit.function.abstract_quantile_function import AbstractQuantileFunction, \
    QuantileFunctionFromMarginFunction, QuantileFunctionFromParamFunction
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.param_function import LinearParamFunction
from extreme_fit.model.margin_model.linear_margin_model.abstract_temporal_linear_margin_model import \
    AbstractTemporalLinearMarginModel, TemporalMarginFitMethod
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.quantile_model.quantile_regression_model import AbstractQuantileRegressionModel
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_quantilreg import ResultFromQuantreg
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class QuantileEstimatorFromMargin(LinearMarginEstimator, AbstractQuantileEstimator):

    def __init__(self, dataset: AbstractDataset, quantile, margin_model_class: type):
        margin_model = margin_model_class(dataset.coordinates, fit_method=TemporalMarginFitMethod.extremes_fevd_mle)
        super().__init__(dataset=dataset, quantile=quantile, margin_model=margin_model)

    @cached_property
    def function_from_fit(self) -> AbstractQuantileFunction:
        linear_margin_function = super().function_from_fit  # type: AbstractMarginFunction
        return QuantileFunctionFromMarginFunction(self.dataset.coordinates, linear_margin_function, self.quantile)
