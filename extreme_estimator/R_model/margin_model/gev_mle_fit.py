import rpy2.robjects as ro
import numpy as np
import rpy2.robjects.numpy2ri as rpyn
import os.path as op

# Defining some constants
from extreme_estimator.R_model.utils import get_associated_r_file


def mle_gev(x_gev: np.ndarray, start_loc=0, start_scale=1, start_shape=0):
    assert np.ndim(x_gev) == 1
    assert start_scale > 0
    r = ro.r
    x_gev = rpyn.numpy2ri(x_gev)
    r.assign('x_gev', x_gev)
    r.assign('python_wrapping', True)
    r.source(file=get_associated_r_file(python_filepath=op.abspath(__file__)))
    res = r.mle_gev(loc=start_loc, scale=start_scale, shape=start_shape)
    mle_params = dict(r.attr(res, 'coef').items())
    return mle_params


class GevMleFit(object):
    GEV_SCALE = 'scale'
    GEV_LOCATION = 'loc'
    GEV_SHAPE = 'shape'

    def __init__(self, x_gev: np.ndarray, start_loc=0, start_scale=1, start_shape=0):
        self.x_gev = x_gev
        self.mle_params = mle_gev(x_gev, start_loc, start_scale, start_shape)
        self.shape = self.mle_params[self.GEV_SHAPE]
        self.scale = self.mle_params[self.GEV_SCALE]
        self.location = self.mle_params[self.GEV_LOCATION]
