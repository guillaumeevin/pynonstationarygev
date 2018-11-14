from extreme_estimator.R_model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)
        assert self.dataset.maxima_gev is not None
        self._margin_function_fitted = None

    @property
    def margin_function_fitted(self):
        assert self._margin_function_fitted is not None, 'Error: estimator has not been not fitted yet'
        return self._margin_function_fitted


class PointWiseMarginEstimator(AbstractMarginEstimator):
    pass


class SmoothMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: AbstractMarginModel):
        super().__init__(dataset)
        assert isinstance(margin_model, AbstractMarginModel)
        self.margin_model = margin_model

    def _fit(self):
        self._margin_function_fitted = self.margin_model.fitmargin_from_maxima_gev(maxima_gev=self.dataset.maxima_gev,
                                                                                   coordinates=self.dataset.coordinates)
