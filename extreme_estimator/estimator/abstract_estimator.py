import time

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset
from spatio_temporal_dataset.slicer.split import Split


class AbstractEstimator(object):
    DURATION = 'Average duration'
    MAE_ERROR = 'Mean Average Error'

    # For each estimator, we shall define:
    # - The loss
    # - The optimization method for each part of the process

    def __init__(self, dataset: AbstractDataset):
        self.dataset = dataset  # type: AbstractDataset
        self.additional_information = dict()
        self._params_fitted = None
        self._margin_function_fitted = None
        self._max_stable_model_fitted = None

    def fit(self):
        ts = time.time()
        self._fit()
        te = time.time()
        self.additional_information[self.DURATION] = int((te - ts) * 1000)

    @property
    def params_fitted(self):
        assert self.is_fitted
        return self._params_fitted

    @property
    def margin_function_fitted(self) -> AbstractMarginFunction:
        assert self.is_fitted
        assert self._margin_function_fitted is not None, 'No margin function has been fitted'
        return self._margin_function_fitted

    def extract_fitted_models_from_fitted_params(self, margin_function_to_fit, full_params_fitted):
        coef_dict = {k: v for k, v in full_params_fitted.items() if 'Coeff' in k}
        self._margin_function_fitted = LinearMarginFunction.from_coef_dict(coordinates=self.dataset.coordinates,
                                                                           gev_param_name_to_linear_dims=margin_function_to_fit.gev_param_name_to_linear_dims,
                                                                           coef_dict=coef_dict)

    @property
    def is_fitted(self):
        return self._params_fitted is not None

    @property
    def train_split(self):
        return self.dataset.train_split

    def scalars(self, true_max_stable_params: dict):
        error = self._error(true_max_stable_params)
        return {**error, **self.additional_information}

    # Methods to override in the child class

    def _fit(self):
        pass

    def _error(self, true_max_stable_params: dict):
        pass
