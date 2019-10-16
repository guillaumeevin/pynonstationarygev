import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams


class GevFit(object):

    def __init__(self, x_gev: np.ndarray, block_size=None):
        assert np.ndim(x_gev) == 1
        assert len(x_gev) > 1
        self.x_gev = x_gev

    @property
    def gev_params(self) -> GevParams:
        raise NotImplementedError