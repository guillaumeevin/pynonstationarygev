import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


def convertFloatVector_to_float(f):
    return np.array(f)[0]


def get_margin_coef_dict(gev_param_name_to_dim, mle_values):
    assert gev_param_name_to_dim is not None
    # Build the Coeff dict from gev_param_name_to_dim
    coef_dict = {}
    i = 0
    for gev_param_name in GevParams.PARAM_NAMES:
        # Add intercept
        intercept_coef_name = LinearCoef.coef_template_str(gev_param_name, LinearCoef.INTERCEPT_NAME).format(1)
        coef_dict[intercept_coef_name] = mle_values[i]
        i += 1
        # Add a potential linear temporal trend
        if gev_param_name in gev_param_name_to_dim:
            temporal_coef_name = LinearCoef.coef_template_str(gev_param_name,
                                                              AbstractCoordinates.COORDINATE_T).format(1)
            coef_dict[temporal_coef_name] = mle_values[i]
            i += 1
    return coef_dict