import os.path as op
import warnings

import numpy as np
import random
import sys
from types import TracebackType
from typing import Dict, Optional

import pandas as pd
import rpy2.robjects as ro
from rpy2 import robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.rinterface._rinterface import RRuntimeError

from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

r = ro.R()
numpy2ri.activate()
pandas2ri.activate()
r.library('SpatialExtremes')


# Notice: R is not reloading all the time, the SpatialExtremes, so it's quite hard to debug or print in the code...
# the best solution for debugging is to copy/paste the code module into a file that belongs to me, and then
# I can put print & stop in the code, and I can understand where are the problems

def set_seed_r(seed=42):
    r("set.seed({})".format(seed))


def get_associated_r_file(python_filepath: str) -> str:
    assert op.exists(python_filepath)
    r_filepath = python_filepath.replace('.py', '.R')
    assert op.exists(r_filepath), r_filepath
    return r_filepath


class WarningMaximumAbsoluteValueTooHigh(Warning):
    pass


def safe_run_r_estimator(function, data, use_start=False, threshold_max_abs_value=100, **parameters):
    # Raise warning if the maximum absolute value is above a threshold
    assert isinstance(data, np.ndarray)
    maximum_absolute_value = np.max(np.abs(data))
    if maximum_absolute_value > threshold_max_abs_value:
        msg = "maxmimum absolute value in data {} is too high, i.e. above the defined threshold {}"\
            .format(maximum_absolute_value, threshold_max_abs_value)
        msg += '\nPotentially in that case, data should be re-normalized'
        warnings.warn(msg, WarningMaximumAbsoluteValueTooHigh)
    parameters['data'] = data
    # First run without using start value
    # Then if it crashes, use start value
    run_successful = False
    while not run_successful:
        current_parameter = parameters.copy()
        if not use_start and 'start' in current_parameter:
            current_parameter.pop('start')
        try:
            res = function(**current_parameter)  # type:
            run_successful = True
        except (RRuntimeError, RRuntimeWarning) as e:
            if not use_start:
                use_start = True
                continue
            elif isinstance(e, RRuntimeError):
                raise Exception('Some R exception have been launched at RunTime: \n {}'.format(e.__repr__()))
            if isinstance(e, RRuntimeWarning):
                print(e.__repr__())
                print('WARNING')
    return res


def retrieve_fitted_values(res: robjects.ListVector) -> Dict[str, float]:
    # todo: maybe if the convergence was not successful I could try other starting point several times
    # Retrieve the resulting fitted values
    fitted_values = res.rx2('fitted.values')
    fitted_values = {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
    return fitted_values


def get_coord(df_coordinates: pd.DataFrame):
    coord = robjects.vectors.Matrix(df_coordinates.values)
    coord.colnames = robjects.StrVector(list(df_coordinates.columns))
    return coord


def get_margin_formula(fit_marge_form_dict) -> Dict:
    return {k: robjects.Formula(v) for k, v in fit_marge_form_dict.items()}

# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
