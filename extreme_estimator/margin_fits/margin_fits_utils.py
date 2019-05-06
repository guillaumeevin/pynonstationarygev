import numpy as np
import rpy2.robjects as ro

from extreme_estimator.extreme_models.utils import r, get_null

"""
These two functions are “extremely light” functions to fit the GEV/GPD. These functions are mainlyuseful 
to compute starting values for the Schlather and Smith mode
If more refined (univariate) analysis have to be performed, users should use more specialised pack-ages
 - e.g. POT, evd, ismev, . . . .
"""


def spatial_extreme_gevmle_fit(x_gev):
    res = r.gevmle(x_gev, method="Nelder")
    return dict(zip(res.names, res))


def spatial_extreme_gpdmle_fit(x_gev, threshold):
    res = r.gpdmle(x_gev, threshold, method="Nelder")
    return dict(zip(res.names, res))


# todo: define more robust function gevmle_fit/gpdmle_fit

"""
IsMev fit functions
"""


def ismev_gev_fit(x_gev, y, mul):
    xdat = ro.FloatVector(x_gev)
    gev_fit = r('gev.fit')
    y = y if y is not None else get_null()
    mul = mul if mul is not None else get_null()
    res = gev_fit(xdat, y, mul)

    # r.assign('python_wrapping', True)
    # r.source(file="""/home/erwan/Documents/projects/spatiotemporalextremes/extreme_estimator/margin_fits/gev/wrapper_ismev_gev_fit.R""")
    # y = np.arange(1, len(x_gev), 1).reshape((-11, 1))
    # res = r.gev_fit(data=xdat, y=y, trend=1)
    return dict(zip(res.names, res))

# if __name__ == '__main__':
#     a = np.array([2, 2])
#     v = ro.vectors.FloatVector((1, 2, 3, 4, 5))
#     ro.globalenv['a'] = a
#     print(r('class(a)'))