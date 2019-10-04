import numpy as np

from extreme_estimator.margin_fits.gev.gev_fit import GevFit
from extreme_estimator.margin_fits.gev.gev_params import GevParams


class IsmevGevFit(GevFit):

    def __init__(self, x_gev: np.ndarray, y=None, mul=None):
        super().__init__(x_gev)
        self.y = y
        self.mul = mul
        self.res = fevd_gev_fit(x_gev, y, mul)

    # @property
    # def gev_params(self) -> GevParams:
    #     assert self.y is None
    #     gev_params_dict = dict(zip(GevParams.PARAM_NAMES, self.res['mle']))
    #     return GevParams.from_dict(gev_params_dict)


def fevd_gev_fit(x, y, mul):
    pass