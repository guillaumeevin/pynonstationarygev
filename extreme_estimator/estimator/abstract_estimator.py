import time

from cached_property import cached_property

from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractEstimator(object):

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset  # type: AbstractDataset
        self._result_from_fit = None  # type: ResultFromFit

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        raise NotImplementedError

    def fit(self):
        raise NotImplementedError

    @property
    def result_from_fit(self) -> ResultFromFit:
        assert self._result_from_fit is not None, 'Fit has not be done'
        return self._result_from_fit

    @cached_property
    def margin_function_fitted(self) -> AbstractMarginFunction:
        return self.extract_function_fitted()

    def extract_function_fitted(self) -> AbstractMarginFunction:
        raise NotImplementedError

    def extract_function_fitted_from_the_model_shape(self, margin_model: LinearMarginModel):
        return LinearMarginFunction.from_coef_dict(coordinates=self.dataset.coordinates,
                                                   gev_param_name_to_dims=margin_model.margin_function_start_fit.gev_param_name_to_dims,
                                                   coef_dict=self.result_from_fit.margin_coef_dict,
                                                   starting_point=margin_model.starting_point)

    #
    @property
    def train_split(self):
        return self.dataset.train_split
