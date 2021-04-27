import numpy as np


def compute_nllh(coordinate_values, maxima_values, margin_function_from_fit, maximum_from_obs=True, assertion_for_inf=True):
    nllh = 0
    for maximum, coordinate in zip(maxima_values, coordinate_values):
        if maximum_from_obs:
            assert len(maximum) == 1, \
                'So far, only one observation for each coordinate, but code would be easy to change'
            maximum = maximum[0]
        gev_params = margin_function_from_fit.get_params(coordinate, is_transformed=True)
        p = gev_params.density(maximum)
        nllh -= np.log(p)
        if assertion_for_inf:
            assert not np.isinf(nllh), '{} {} {}'.format(gev_params, coordinate, maximum)
    return nllh

