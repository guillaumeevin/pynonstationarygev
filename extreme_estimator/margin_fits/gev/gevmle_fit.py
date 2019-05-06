import numpy as np

from extreme_estimator.margin_fits.gev.gev_fit import GevFit
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from extreme_estimator.margin_fits.margin_fits_utils import spatial_extreme_gevmle_fit


class GevMleFit(GevFit):

    def __init__(self, x_gev: np.ndarray, block_size=None):
        super().__init__(x_gev, block_size)
        self._gev_params = spatial_extreme_gevmle_fit(x_gev)
        self.gev_params_object = GevParams.from_dict({**self._gev_params, 'block_size': block_size})

    @property
    def gev_params(self):
        return self._gev_params


