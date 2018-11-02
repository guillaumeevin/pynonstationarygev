import rpy2.robjects as ro
import pandas as pd
import numpy as np
import rpy2.robjects.numpy2ri as npr


def get_loaded_r():
    r = ro.r
    ro.numpy2ri.activate()
    r.library('SpatialExtremes')
    return r


# def conversion_to_FloatVector(data):
#     """Convert DataFrame or numpy array into FloatVector for r"""
#     if isinstance(data, pd.DataFrame):
#         data = data.values
#     assert isinstance(data, np.ndarray)
#     return npr.numpy2ri(data)
