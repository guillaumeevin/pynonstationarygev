from typing import Union

import numpy as np

from extreme_data.eurocode_data.utils import EUROCODE_QUANTILE
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.model.result_from_model_fit.result_from_extremes.abstract_extract_eurocode_return_level import \
    ExtractEurocodeReturnLevelFromMyBayesianExtremes, ExtractEurocodeReturnLevelFromCiMethod
from extreme_fit.model.result_from_model_fit.result_from_extremes.confidence_interval_method import \
    ConfidenceIntervalMethodFromExtremes


class EurocodeConfidenceIntervalFromExtremes(object):
    quantile_level = EUROCODE_QUANTILE

    def __init__(self, mean_estimate, confidence_interval):
        self.mean_estimate = mean_estimate
        self.confidence_interval = confidence_interval

    @property
    def interval_size(self):
        return self.confidence_interval[1] - self.confidence_interval[0]

    @property
    def triplet(self):
        return self.confidence_interval[0], self.mean_estimate, self.confidence_interval[1]

    @classmethod
    def from_estimator_extremes(cls, estimator_extremes: LinearMarginEstimator,
                                ci_method: ConfidenceIntervalMethodFromExtremes,
                                coordinate: Union[int, np.ndarray]):

        if ci_method == ConfidenceIntervalMethodFromExtremes.my_bayes:
            extractor = ExtractEurocodeReturnLevelFromMyBayesianExtremes(estimator_extremes, ci_method, coordinate, cls.quantile_level)
        else:
            extractor = ExtractEurocodeReturnLevelFromCiMethod(estimator_extremes, ci_method, coordinate, cls.quantile_level)
        return cls(extractor.mean_estimate,  extractor.confidence_interval)




