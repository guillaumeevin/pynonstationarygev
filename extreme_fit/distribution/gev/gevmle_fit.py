import numpy as np

from extreme_fit.distribution.gev.gev_fit import GevFit
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.utils import r

"""
These two functions are “extremely light” functions to fit the GEV/GPD. These functions are mainlyuseful 
to compute starting values for the Schlather and Smith mode
If more refined (univariate) analysis have to be performed, users should use more specialised pack-ages
 - e.g. POT, evd, ismev, . . . .
"""


def spatial_extreme_gevmle_fit(x_gev):
    res = r.gevmle(x_gev, method="Nelder")
    return dict(zip(res.names, res))


class GevMleFit(GevFit):

    def __init__(self, x_gev: np.ndarray, block_size=None):
        super().__init__(x_gev, block_size)
        self._gev_params = spatial_extreme_gevmle_fit(x_gev)
        self.gev_params_object = GevParams.from_dict({**self._gev_params, 'block_size': block_size})

    @property
    def gev_params(self) -> GevParams:
        return self.gev_params_object
