import copy
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
r.library('quantreg')
# Desactivate temporarily warnings
default_filters = warnings.filters.copy()
warnings.filterwarnings("ignore")
# Load ismev
r.library('ismev')
# Load fevd fixed
for j, filename in enumerate(['ci_fevd_fixed.R', 'fevd_fixed.R', 'summary_fevd_fixed.R', 'gnfit_fixed.R']):
    folder = 'gev' if j <= 2 else "gumbel"
    fevd_fixed_filepath = op.join(get_root_path(), 'extreme_fit', 'distribution', folder, filename)
    assert op.exists(fevd_fixed_filepath)
    r.source(fevd_fixed_filepath)
# Reactivate warning
warnings.filters = default_filters


# Notice: R is not reloading all the time, the SpatialExtremes, so it's quite hard to debug or print in the code...
# the best solution for debugging is to copy/paste the code module into a file that belongs to me, and then
# I can put print & stop in the code, and I can understand where are the problems

def set_seed_for_test(seed=42):
    set_seed_r(seed=seed)
    np.random.seed(seed=seed)
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


class WarningTooMuchZeroValues(Warning):
    pass


class WarningMaximumAbsoluteValueTooHigh(Warning):
    pass


class OptimizationConstants(object):
    USE_MAXIT = False


class SafeRunException(Exception):
    pass


def safe_run_r_estimator(function, data=None, start_dict=None, max_ratio_between_two_extremes_values=10, maxit=1000000,
                         nb_tries_for_start_value=5, **parameters) -> robjects.ListVector:
    try:
        return _safe_run_r_estimator(function, data, max_ratio_between_two_extremes_values, maxit, **parameters)
    except SafeRunException as e:
        if start_dict is not None:
            for _ in range(nb_tries_for_start_value):
                parameters['start'] = r.list(**start_dict)
                try:
                    return _safe_run_r_estimator(function, data, max_ratio_between_two_extremes_values, maxit,
                                                 **parameters)
                except Exception:
                    continue
        else:
            raise e


def _safe_run_r_estimator(function, data=None, max_ratio_between_two_extremes_values=10, maxit=1000000,
                          **parameters) -> robjects.ListVector:
    if OptimizationConstants.USE_MAXIT:
        # Add optimization parameters
        optim_dict = {'maxit': maxit}
        parameters['control'] = r.list(**optim_dict)

    # Raise error if needed
    if 'x' in parameters and np.isnan(parameters['x']).any():
        raise ValueError('x contains NaN values')

    # Some checks for Spatial Extremes
    if data is not None:
        # Raise some warnings
        if isinstance(data, np.ndarray):
            # Raise warning if the gap is too important between the two biggest values of data
            sorted_data = sorted(data.flatten())
            if sorted_data[-2] * max_ratio_between_two_extremes_values < sorted_data[-1]:
                msg = "maxmimum absolute value in data {} is too high, i.e. above the defined threshold {}" \
                    .format(sorted_data[-1], max_ratio_between_two_extremes_values)
                msg += '\nPotentially in that case, data should be re-normalized'
                warnings.warn(msg, WarningMaximumAbsoluteValueTooHigh)
            # Raise warning if ratio of zeros in data is above some percentage (90% so far)
            limit_percentage = 90
            if 100 * np.count_nonzero(data) / len(data) < limit_percentage:
                msg = 'data contains more than {}% of zero values'.format(100 - limit_percentage)
                warnings.warn(msg, WarningTooMuchZeroValues)
        # Add data to the parameters
        parameters['data'] = data

    run_successful = False
    res = None
    f = io.StringIO()
    # Warning print will not work in this part
    with redirect_stdout(f):
        while not run_successful:
            try:
                res = function(**parameters)  # type:
                run_successful = True
            except (RRuntimeError, RRuntimeWarning) as e:
                if isinstance(e, RRuntimeError):
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
    colname = df_coordinates.columns
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


def new_coef_name_to_old_coef_names():
    return {
        'location.fun': ['loc.form', 'temp.form.loc'],
        'scale.fun': ['scale.form', 'temp.form.scale'],
        'shape.fun': ['shape.form', 'temp.form.shape'],
    }


def get_margin_formula_extremes(fit_marge_form_dict) -> Dict:
    v_to_str = lambda v: ' '.join(v.split()[2:]) if v != 'NULL' else ' 1'
    form_dict = {
        k: '~ ' + ' + '.join(
            [v_to_str(fit_marge_form_dict[e]) for e in l if e in fit_marge_form_dict])
        for k, l in new_coef_name_to_old_coef_names().items()
    }
    return {k: robjects.Formula(v) for k, v in form_dict.items()}

# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
