import os.path as op
import rpy2.robjects as ro
import pandas as pd
import numpy as np
import rpy2.robjects.numpy2ri as npr


def get_loaded_r() -> ro.R:
    r = ro.r
    ro.numpy2ri.activate()
    r.library('SpatialExtremes')
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
