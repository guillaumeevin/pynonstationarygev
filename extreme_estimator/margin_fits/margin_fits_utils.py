from extreme_estimator.extreme_models.utils import r

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
