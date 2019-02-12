from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.extreme_params import ExtremeParams


class GevParams(ExtremeParams):

    # Parameters
    PARAM_NAMES = [ExtremeParams.LOC, ExtremeParams.SCALE, ExtremeParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + ExtremeParams.QUANTILE_NAMES

    def __init__(self, loc: float, scale: float, shape: float, block_size: int = None):
        super().__init__(loc, scale, shape)
        self.block_size = block_size

    def quantile(self, p) -> float:
        return r.qgev(p, self.location, self.scale, self.shape)[0]

    @property
    def param_values(self):
        return [self.location, self.scale, self.shape]