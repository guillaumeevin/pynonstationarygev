import numpy as np

from extreme_fit.distribution.gpd.gpd_params import GpdParams
from extreme_fit.model.utils import r

"""
These two functions are “extremely light” functions to fit the GEV/GPD. These functions are mainlyuseful 
to compute starting values for the Schlather and Smith mode
If more refined (univariate) analysis have to be performed, users should use more specialised pack-ages
 - e.g. POT, evd, ismev, . . . .
"""


def spatial_extreme_gpdmle_fit(x_gev, threshold):
    res = r.gpdmle(x_gev, threshold, method="Nelder")
    return dict(zip(res.names, res))

class GpdMleFit(object):

    def __init__(self, x_gev: np.ndarray, threshold):
        assert np.ndim(x_gev) == 1
        assert len(x_gev) > 1
        self.x_gev = x_gev
        self.threshold = threshold
        self.mle_params = spatial_extreme_gpdmle_fit(x_gev, threshold)
        self.gpd_params = GpdParams.from_dict({**self.mle_params, 'threshold': threshold})