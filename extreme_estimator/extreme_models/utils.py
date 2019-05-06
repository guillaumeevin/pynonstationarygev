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

from extreme_estimator.extreme_models.result_from_fit import ResultFromFit, ResultFromSpatialExtreme

r = ro.R()
numpy2ri.activate()
pandas2ri.activate()
r.library('SpatialExtremes')
r.library('ismev')


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


def safe_run_r_estimator(function, data=None, use_start=False, threshold_max_abs_value=100, **parameters) -> robjects.ListVector:
    # Some checks for Spatial Extremes
    if data is not None:
        # Raise warning if the maximum absolute value is above a threshold
        assert isinstance(data, np.ndarray)
        maximum_absolute_value = np.max(np.abs(data))
        if maximum_absolute_value > threshold_max_abs_value:
            msg = "maxmimum absolute value in data {} is too high, i.e. above the defined threshold {}" \
                .format(maximum_absolute_value, threshold_max_abs_value)
            msg += '\nPotentially in that case, data should be re-normalized'
            warnings.warn(msg, WarningMaximumAbsoluteValueTooHigh)
        parameters['data'] = data
    # First run without using start value
    # Then if it crashes, use start value
    run_successful = False
    res = None
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


def get_coord(df_coordinates: pd.DataFrame):
    coord = robjects.vectors.Matrix(df_coordinates.values)
    coord.colnames = robjects.StrVector(list(df_coordinates.columns))
    return coord


def get_null():
    as_null = r['as.null']
    return as_null(1.0)


def get_margin_formula(fit_marge_form_dict) -> Dict:
    margin_formula = {k: robjects.Formula(v) if v != 'NULL' else get_null() for k, v in fit_marge_form_dict.items()}
    return margin_formula

# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
