from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.extreme_params import ExtremeParams


class GpdParams(ExtremeParams):

    def __init__(self, loc: float, scale: float, shape: float, threshold: float = None):
        super().__init__(loc, scale, shape)
        self.threshold = threshold

    def quantile(self, p) -> float:
        return r.qgpd(p, self.location, self.scale, self.shape)[0]
