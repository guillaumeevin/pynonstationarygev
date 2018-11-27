import os.path as op
import random
import sys

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri


def get_loaded_r() -> ro.R:
    r = ro.r
    numpy2ri.activate()
    pandas2ri.activate()
    r.library('SpatialExtremes')
    # max_int = r('.Machine$integer.max')
    # seed = random.randrange(max_int)
    # r("set.seed({})".format(seed))
    return r


def get_associated_r_file(python_filepath: str) -> str:
    assert op.exists(python_filepath)
    r_filepath = python_filepath.replace('.py', '.R')
    assert op.exists(r_filepath), r_filepath
    return r_filepath

# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
