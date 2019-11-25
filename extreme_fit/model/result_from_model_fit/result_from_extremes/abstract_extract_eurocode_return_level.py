from typing import List

import numpy as np
from cached_property import cached_property

from experiment.eurocode_data.utils import EUROCODE_QUANTILE, YEAR_OF_INTEREST_FOR_RETURN_LEVEL
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.estimator.margin_estimator.abstract_margin_estimator import LinearMarginEstimator
from extreme_fit.estimator.utils import load_margin_function
from extreme_fit.model.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    ResultFromBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_mle_extremes import ResultFromMleExtremes
from root_utils import classproperty


class AbstractExtractEurocodeReturnLevel(object):
    ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY = 0.05

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


class ExtractEurocodeReturnLevelFromCiMethod(AbstractExtractEurocodeReturnLevel):
    result_from_fit: ResultFromMleExtremes

    @property
    def transformed_temporal_covariate(self):
        return self.estimator.dataset.coordinates.transformation.transform_float(self.temporal_covariate)

    @cached_property
    def confidence_interval_method(self):
        return self.result_from_fit.confidence_interval_method(self.quantile_level,
                                                               self.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY,
                                                               self.transformed_temporal_covariate,
                                                               self.ci_method)

    @property
    def mean_estimate(self) -> np.ndarray:
        return self.confidence_interval_method[0]

    @property
    def confidence_interval(self):
        return self.confidence_interval_method[1]


class ExtractEurocodeReturnLevelFromMyBayesianExtremes(AbstractExtractEurocodeReturnLevel):
    result_from_fit: ResultFromBayesianExtremes

    def __init__(self, estimator: LinearMarginEstimator, ci_method, temporal_covariate, quantile_level=EUROCODE_QUANTILE):
        super().__init__(estimator, ci_method, temporal_covariate, quantile_level)
        assert isinstance(self.result_from_fit, ResultFromBayesianExtremes)

    @property
    def margin_functions_from_fit(self) -> List[LinearMarginFunction]:
        margin_functions = []
        for _, s in self.result_from_fit.df_posterior_samples.iterrows():
            coef_dict = self.result_from_fit.get_coef_dict_from_posterior_sample(s)
            margin_function = load_margin_function(self.estimator, self.estimator.margin_model, coef_dict=coef_dict)
            margin_functions.append(margin_function)
        return margin_functions

    @property
    def gev_params_from_fit_for_temporal_covariate(self) -> List[GevParams]:
        return [margin_function.get_gev_params(coordinate=np.array([self.temporal_covariate]), is_transformed=False)
                for margin_function in self.margin_functions_from_fit]

    @cached_property
    def posterior_eurocode_return_level_samples_for_temporal_covariate(self) -> np.ndarray:
        return np.array(
            [p.quantile(self.quantile_level) for p in self.gev_params_from_fit_for_temporal_covariate])

    @property
    def mean_estimate(self) -> np.ndarray:
        # Mean posterior value here
        return np.mean(self.posterior_eurocode_return_level_samples_for_temporal_covariate)

    @property
    def confidence_interval(self):
        # Bottom and upper quantile correspond to the quantile
        bottom_quantile = self.ALPHA_CONFIDENCE_INTERVAL_UNCERTAINTY / 2
        bottom_and_upper_quantile = (bottom_quantile, 1 - bottom_quantile)
        return [np.quantile(self.posterior_eurocode_return_level_samples_for_temporal_covariate, q=q)
                for q in bottom_and_upper_quantile]
