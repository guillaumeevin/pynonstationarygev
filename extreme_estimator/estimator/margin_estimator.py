from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.smooth_margin_model import LinearMarginModel
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)
        assert self.dataset.maxima_gev() is not None
        self._margin_function_fitted = None

    @property
    def margin_function_fitted(self) -> AbstractMarginFunction:
        assert self._margin_function_fitted is not None, 'Error: estimator has not been fitted'
        return self._margin_function_fitted


class PointWiseMarginEstimator(AbstractMarginEstimator):
    pass


class ConstantMarginEstimator(AbstractMarginEstimator):

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel):
        super().__init__(dataset)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self):
        self._margin_function_fitted = self.margin_model.margin_function_start_fit


class SmoothMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: LinearMarginModel):
        super().__init__(dataset)
        assert isinstance(margin_model, LinearMarginModel)
        self.margin_model = margin_model

    def _fit(self):
        maxima_gev = self.dataset.maxima_gev(split=self.train_split)
        coordinate_values = self.dataset.coordinates_values(self.train_split)
        self._margin_function_fitted = self.margin_model.fitmargin_from_maxima_gev(maxima_gev=maxima_gev,
                                                                                   coordinates_values=coordinate_values)
