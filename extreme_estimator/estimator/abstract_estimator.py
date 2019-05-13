import time

from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractEstimator(object):
    DURATION = 'Average duration'
    MAE_ERROR = 'Mean Average Error'

    # For each estimator, we shall define:
    # - The loss
    # - The optimization method for each part of the process

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset  # type: AbstractDataset
        self.additional_information = dict()
        self._result_from_fit = None  # type: ResultFromFit
        self._margin_function_fitted = None
        self._max_stable_model_fitted = None

    @classmethod
    def from_dataset(cls, dataset: AbstractDataset):
        # raise NotImplementedError('from_dataset class constructor has not been implemented for this class')
        pass

    def fit(self):
        ts = time.time()
        self._fit()
        te = time.time()
        self.additional_information[self.DURATION] = int((te - ts) * 1000)

    @property
    def result_from_fit(self) -> ResultFromFit:
        assert self._result_from_fit is not None, 'Fit has not be done'
        return self._result_from_fit

    @property
    def margin_function_fitted(self) -> AbstractMarginFunction:
        if self._margin_function_fitted is None:
            self._margin_function_fitted = self.extract_function_fitted()
        assert self._margin_function_fitted is not None, 'No margin function has been fitted'
        return self._margin_function_fitted

    def extract_function_fitted(self) -> AbstractMarginFunction:
        raise NotImplementedError

    def extract_function_fitted_from_the_model_shape(self, margin_model: LinearMarginModel):
        return LinearMarginFunction.from_coef_dict(coordinates=self.dataset.coordinates,
                                                   gev_param_name_to_dims=margin_model.margin_function_start_fit.gev_param_name_to_dims,
                                                   coef_dict=self.result_from_fit.margin_coef_dict,
                                                   starting_point=margin_model.transformed_starting_point)

    # @property
    # def max_stable_fitted(self) -> AbstractMarginFunction:
    #     assert self._margin_function_fitted is not None, 'Error: estimator has not been fitted'
    #     return self._margin_function_fitted

    @property
    def train_split(self):
        return self.dataset.train_split

    # Methods to override in the child class

    def _fit(self):
        raise NotImplementedError
