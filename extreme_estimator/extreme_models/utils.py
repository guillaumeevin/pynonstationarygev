import os.path as op
import random
import sys

import rpy2.robjects as ro

from rpy2.robjects import numpy2ri
from rpy2.robjects import pandas2ri

r = ro.R()
numpy2ri.activate()
pandas2ri.activate()
r.library('SpatialExtremes')


def set_seed_r(seed=42):
    r("set.seed({})".format(seed))


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
