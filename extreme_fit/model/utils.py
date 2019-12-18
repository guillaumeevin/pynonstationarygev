import io

import os.path as op
import random
import warnings
from contextlib import redirect_stdout
from typing import Dict

import numpy as np
import pandas as pd
import rpy2.robjects as ro
from rpy2 import robjects
from rpy2.rinterface import RRuntimeWarning
from rpy2.rinterface._rinterface import RRuntimeError
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

# Load R variables
from root_utils import get_root_path

r = ro.R()
numpy2ri.activate()
pandas2ri.activate()
r.library('SpatialExtremes')
r.library('data.table')
# Desactivate temporarily warnings
default_filters = warnings.filters.copy()
warnings.filterwarnings("ignore")
# Load ismev
r.library('ismev')
# Load fevd fixed
for filename in ['ci_fevd_fixed.R', 'fevd_fixed.R']:
    fevd_fixed_filepath = op.join(get_root_path(), 'extreme_fit', 'distribution', 'gev', filename)
    assert op.exists(fevd_fixed_filepath)
    r.source(fevd_fixed_filepath)
# Reactivate warning
warnings.filters = default_filters


# Notice: R is not reloading all the time, the SpatialExtremes, so it's quite hard to debug or print in the code...
# the best solution for debugging is to copy/paste the code module into a file that belongs to me, and then
# I can put print & stop in the code, and I can understand where are the problems

def set_seed_for_test(seed=42):
    set_seed_r(seed=seed)
    random.seed(seed)


def set_seed_r(seed=42):
    r("set.seed({})".format(seed))


def get_associated_r_file(python_filepath: str) -> str:
    assert op.exists(python_filepath)
    r_filepath = python_filepath.replace('.py', '.R')
    assert op.exists(r_filepath), r_filepath
    return r_filepath


class WarningWhileRunningR(Warning):
    pass


class WarningMaximumAbsoluteValueTooHigh(Warning):
    pass


class OptimizationConstants(object):
    USE_MAXIT = False


class SafeRunException(Exception):
    pass


def safe_run_r_estimator(function, data=None, use_start=False, threshold_max_abs_value=100, maxit=1000000,
                         **parameters) -> robjects.ListVector:
    if OptimizationConstants.USE_MAXIT:
        # Add optimization parameters
        optim_dict = {'maxit': maxit}
        parameters['control'] = r.list(**optim_dict)

    # Some checks for Spatial Extremes
    if data is not None:
        # Raise warning if the maximum absolute value is above a threshold
        if isinstance(data, np.ndarray):
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
    f = io.StringIO()
    # Warning print will not work in this part
    with redirect_stdout(f):
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
                    raise SafeRunException('Some R exception have been launched at RunTime: \n {}'.format(e.__repr__()))
                if isinstance(e, RRuntimeWarning):
                    warnings.warn(e.__repr__(), WarningWhileRunningR)
    return res


def get_coord(df_coordinates: pd.DataFrame):
    coord = robjects.vectors.Matrix(df_coordinates.values)
    coord.colnames = robjects.StrVector(list(df_coordinates.columns))
    return coord


def get_coord_df(df_coordinates: pd.DataFrame):
    coord = pandas2ri.py2ri_pandasdataframe(df_coordinates)
    # coord = r.transpose(coord)
    colname = df_coordinates.columns[0]
    coord.colnames = r.c(colname)
    coord = r('data.frame')(coord, stringsAsFactors=True)
    return coord


def get_null():
    as_null = r['as.null']
    return as_null(1.0)


# todo: move that to the result class maybe
def get_margin_formula_spatial_extreme(fit_marge_form_dict) -> Dict:
    margin_formula = {k: robjects.Formula(v) if v != 'NULL' else get_null() for k, v in fit_marge_form_dict.items()}
    return margin_formula


def old_coef_name_to_new_coef_name():
    return {
        'temp.form.loc': 'location.fun',
        'temp.form.scale': 'scale.fun',
        'temp.form.shape': 'shape.fun',
    }


def get_margin_formula_extremes(fit_marge_form_dict) -> Dict:
    d = old_coef_name_to_new_coef_name()
    form_dict = {d[k]: ' '.join(v.split()[1:]) if v != 'NULL' else ' ~1' for k, v in fit_marge_form_dict.items()}
    return {k: robjects.Formula(v) for k, v in form_dict.items()}

# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
