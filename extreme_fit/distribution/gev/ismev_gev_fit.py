import numpy as np

from extreme_fit.distribution.gev.gev_fit import GevFit
from extreme_fit.distribution.gev.gev_params import GevParams
import rpy2.robjects as ro

from extreme_fit.model.utils import r, get_null


"""
IsMev fit functions
"""


def ismev_gev_fit(x_gev, y, mul):
    """
    For non-stationary fitting it is recommended that the covariates within the generalized linear models are
    (at least approximately) centered and scaled (i.e.the columns of ydat should be approximately centered and scaled).
    """
    xdat = ro.FloatVector(x_gev)
    gev_fit = r('gev.fit')
    y = y if y is not None else get_null()
    mul = mul if mul is not None else get_null()
    res = gev_fit(xdat, y, mul)

    # r.assign('python_wrapping', True)
    # r.source(file="""/home/erwan/Documents/projects/spatiotemporalextremes/extreme_fit/distribution/gev/wrapper_ismev_gev_fit.R""")
    # y = np.arange(1, len(x_gev), 1).reshape((-11, 1))
    # res = r.gev_fit(data=xdat, y=y, trend=1)
    return dict(zip(res.names, res))

class IsmevGevFit(GevFit):
    # todo: this could be modeled with a margin_function depending only on time
    # todo: I should remove the call to this object, in order to centralize all the calls

    def __init__(self, x_gev: np.ndarray, y=None, mul=None):
        super().__init__(x_gev)
        self.y = y
        self.mul = mul
        self.res = ismev_gev_fit(x_gev, y, mul)

    @property
    def gev_params(self) -> GevParams:
        assert self.y is None
        gev_params_dict = dict(zip(GevParams.PARAM_NAMES, self.res['mle']))
        return GevParams.from_dict(gev_params_dict)



