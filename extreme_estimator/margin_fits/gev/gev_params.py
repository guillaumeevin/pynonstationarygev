from extreme_estimator.extreme_models.utils import r
from extreme_estimator.margin_fits.extreme_params import ExtremeParams
import numpy as np


class GevParams(ExtremeParams):

    # Parameters
    PARAM_NAMES = [ExtremeParams.LOC, ExtremeParams.SCALE, ExtremeParams.SHAPE]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + ExtremeParams.QUANTILE_NAMES

    def __init__(self, loc: float, scale: float, shape: float, block_size: int = None):
        super().__init__(loc, scale, shape)
        self.block_size = block_size

    def quantile(self, p) -> float:
        if self.has_undefined_parameters:
            return np.nan
        else:
            return r.qgev(p, self.location, self.scale, self.shape)[0]

    @property
    def param_values(self):
        if self.has_undefined_parameters:
            return [np.nan for _ in range(3)]
        else:
            return [self.location, self.scale, self.shape]