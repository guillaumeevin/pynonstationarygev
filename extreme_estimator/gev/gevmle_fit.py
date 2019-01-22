import rpy2.robjects as ro
import numpy as np
import rpy2.robjects.numpy2ri as rpyn
import os.path as op

# Defining some constants
from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.utils import get_associated_r_file

from extreme_estimator.extreme_models.utils import r


# def gev_mle_fit(x_gev: np.ndarray, start_loc=0.0, start_scale=1.0, start_shape=1.0):
#     assert np.ndim(x_gev) == 1
#     assert start_scale > 0
#     r = ro.r
#     x_gev = rpyn.numpy2ri(x_gev)
#     r.assign('x_gev', x_gev)
#     r.assign('python_wrapping', True)
#     r.source(file=get_associated_r_file(python_filepath=op.abspath(__file__)))
#     print(start_loc, start_scale, start_shape)
#     res = r.mle_gev(loc=start_loc, scale=start_scale, shape=start_shape)
#     mle_params = dict(r.attr(res, 'coef').items())
#     print('mle params', mle_params)
#     return mle_params

def spatial_extreme_gevmle_fit(x_gev):
    res = r.gevmle(x_gev, method="Nelder")
    return dict(zip(res.names, res))


class GevMleFit(object):

    def __init__(self, x_gev: np.ndarray):
        self.x_gev = x_gev
        self.mle_params = spatial_extreme_gevmle_fit(x_gev)
        self.shape = self.mle_params[GevParams.GEV_SHAPE]
        self.scale = self.mle_params[GevParams.GEV_SCALE]
        self.loc = self.mle_params[GevParams.GEV_LOC]
