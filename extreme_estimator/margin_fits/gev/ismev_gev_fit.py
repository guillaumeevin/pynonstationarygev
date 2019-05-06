import numpy as np

from extreme_estimator.margin_fits.gev.gev_fit import GevFit
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.margin_fits_utils import spatial_extreme_gevmle_fit, ismev_gev_fit


class IsmevGevFit(GevFit):
    # todo: this could be modeled with a margin_function depending only on time

    def __init__(self, x_gev: np.ndarray, y=None, mul=None):
        super().__init__(x_gev)
        self.y = y
        self.mul = mul
        self.res = ismev_gev_fit(x_gev, y, mul)

    @property
    def gev_params(self):
        assert self.y is None
        return dict(zip(GevParams.PARAM_NAMES, self.res['mle']))



