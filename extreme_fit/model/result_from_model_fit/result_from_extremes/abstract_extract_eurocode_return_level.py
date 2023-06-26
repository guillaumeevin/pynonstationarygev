from typing import List

import numpy as np
from cached_property import cached_property

from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    ResultFromBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_mle_extremes import ResultFromMleExtremes
from root_utils import classproperty

class AbstractExtractEurocodeReturnLevel(object):
    ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.05
    NB_BOOTSTRAP = 1000

    @classproperty
    def bottom_and_upper_quantile(cls):
        bottom_quantile = cls.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY / 2
        bottom_and_upper_quantile = (bottom_quantile, 1 - bottom_quantile)
        return bottom_and_upper_quantile

    def __init__(self, estimator: LinearMarginEstimator, ci_method, temporal_covariate, quantile_level=EUROCODE_QUANTILE):
        self.ci_method = ci_method
        self.estimator = estimator
        self.result_from_fit = self.estimator.result_from_model_fit
        self.temporal_covariate = temporal_covariate
        # Fixed Parameters
        self.quantile_level = quantile_level

    @classproperty
    def percentage_confidence_interval(cls) -> int:
        return int(100 * (1 - cls.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY))

    @property
    def mean_estimate(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def confidence_interval(self):
        raise NotImplementedError





