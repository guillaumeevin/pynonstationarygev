import numpy as np

from extreme_fit.distribution.abstract_extreme_params import AbstractExtremeParams


def nan_if_undefined_wrapper(func):
    def wrapper(obj: AbstractExtremeParams, *args, **kwargs):
        if obj.has_undefined_parameters:
            return np.nan
        return func(obj, *args, **kwargs)

    return wrapper
