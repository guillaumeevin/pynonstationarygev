from multiprocessing import Pool

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from root_utils import batch_nb_cores, batch


def compute_nllh(coordinate_values, maxima_values, margin_function_from_fit,
                 maximum_from_obs=True, assertion_for_inf=True, gumbel_standardization=False):
    list_of_pair = list(zip(maxima_values, coordinate_values))
    args = assertion_for_inf, list_of_pair, margin_function_from_fit, maximum_from_obs, gumbel_standardization
    return compute_nllh_for_list_of_pair(args)


class NllhIsInfException(Exception):
    pass


def compute_nllh_for_list_of_pair(args):
    assertion_for_inf, list_of_pair, margin_function_from_fit, maximum_from_obs, gumbel_standardization = args
    nllh = 0
    for maximum, coordinate in list_of_pair:
        if maximum_from_obs:
            assert len(maximum) == 1, \
                'So far, only one observation for each coordinate, but code would be easy to change'
            maximum = maximum[0]
        gev_params = margin_function_from_fit.get_params(coordinate)
        if gumbel_standardization:
            gev_params_gumbel = GevParams(0, 1, 0)
            p = gev_params_gumbel.density(gev_params.gumbel_standardization(maximum))
        else:
            p = gev_params.density(maximum)
        nllh -= np.log(p)
        if assertion_for_inf:
            if np.isinf(nllh):
                msg = '{} {} {}'.format(gev_params, coordinate, maximum)
                raise NllhIsInfException(msg)
    return nllh
