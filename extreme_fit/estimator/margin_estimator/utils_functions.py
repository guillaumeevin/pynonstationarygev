from multiprocessing import Pool

import numpy as np

from root_utils import batch_nb_cores, batch


def compute_nllh(coordinate_values, maxima_values, margin_function_from_fit,
                 maximum_from_obs=True, assertion_for_inf=True):
    list_of_pair = list(zip(maxima_values, coordinate_values))
    args = assertion_for_inf, list_of_pair, margin_function_from_fit, maximum_from_obs
    return compute_nllh_for_list_of_pair(args)


class NllhIsInfException(Exception):
    pass


def compute_nllh_for_list_of_pair(args):
    assertion_for_inf, list_of_pair, margin_function_from_fit, maximum_from_obs = args
    nllh = 0
    for maximum, coordinate in list_of_pair:
        if maximum_from_obs:
            assert len(maximum) == 1, \
                'So far, only one observation for each coordinate, but code would be easy to change'
            maximum = maximum[0]
        gev_params = margin_function_from_fit.get_params(coordinate, is_transformed=True)
        p = gev_params.density(maximum)
        nllh -= np.log(p)
        if assertion_for_inf:
            if np.isinf(nllh):
                msg = '{} {} {}'.format(gev_params, coordinate, maximum)
                raise NllhIsInfException(msg)
    return nllh


def compute_nllh_with_multiprocessing_for_large_samples(coordinate_values, maxima_values, margin_function_from_fit,
                                                        maximum_from_obs=True,
                                                        assertion_for_inf=True):
    if len(coordinate_values) > 2000:
        return compute_nllh_with_multiprocessing(coordinate_values, maxima_values, margin_function_from_fit,
                                                 maximum_from_obs,
                                                 assertion_for_inf)
    else:
        return compute_nllh(coordinate_values, maxima_values, margin_function_from_fit, maximum_from_obs,
                            assertion_for_inf)


def compute_nllh_with_multiprocessing(coordinate_values, maxima_values, margin_function_from_fit, maximum_from_obs=True,
                                      assertion_for_inf=True):
    list_of_pair = list(zip(maxima_values, coordinate_values))
    nb_cores = 7
    batch_list_of_pair = batch_nb_cores(list_of_pair, nb_cores)
    list_of_args = [(assertion_for_inf, list_of_pair, margin_function_from_fit, maximum_from_obs)
                    for list_of_pair in batch_list_of_pair]
    with Pool(nb_cores) as p:
        result_list = p.map(compute_nllh_for_list_of_pair, list_of_args)
        nllh = sum(result_list)
    return nllh
