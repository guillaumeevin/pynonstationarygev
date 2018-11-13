from extreme_estimator.R_fit.gev_fit.abstract_margin_model import AbstractMarginModel
from extreme_estimator.estimator.abstract_estimator import AbstractEstimator
from spatio_temporal_dataset.dataset.abstract_dataset import AbstractDataset


class AbstractMarginEstimator(AbstractEstimator):

    def __init__(self, dataset: AbstractDataset):
        super().__init__(dataset)
        assert self.dataset.maxima_gev is not None


class PointWiseMarginEstimator(AbstractMarginEstimator):
    pass


class SmoothMarginEstimator(AbstractMarginEstimator):
    """# with different type of marginals: cosntant, linear...."""

    def __init__(self, dataset: AbstractDataset, margin_model: AbstractMarginModel):
        super().__init__(dataset)
        self.margin_model = margin_model
        self.df_gev_params = None

    def _fit(self):
        self.df_gev_params = self.margin_model.fitmargin(maxima=self.dataset.maxima_gev,
                                                         coord=self.dataset.coordinates)
