from collections import OrderedDict

import numpy as np

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.param_function.linear_coef import LinearCoef
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates



def convertFloatVector_to_float(f):
    return np.array(f)[0]


def get_margin_coef_ordered_dict(param_name_to_dims, mle_values, type_for_mle="GEV", dim_to_coordinate_name=None,
                                 param_name_to_name_of_the_climatic_effects=None,
                                 linear_effects=(False, False, False)):
    assert param_name_to_dims is not None
    # Build the Coeff dict from param_name_to_dim
    coef_dict = OrderedDict()
    i = 0
    for j, param_name in enumerate(GevParams.PARAM_NAMES):
        # Add intercept (i.e. stationary parameter)
        intercept_coef_name = LinearCoef.coef_template_str(param_name, LinearCoef.INTERCEPT_NAME).format(1)
        if type_for_mle == "Gumbel" and param_name == GevParams.SHAPE:
            coef_value = 0
        else:
            coef_value = mle_values[i]
        coef_dict[intercept_coef_name] = coef_value
        i += 1
        # Add a potential linear temporal trend
        if param_name in param_name_to_dims:
            dims = param_name_to_dims[param_name]
            if isinstance(dims[0], int):
                coef_name = LinearCoef.coef_template_str(param_name, AbstractCoordinates.COORDINATE_T).format(1)
                coef_dict[coef_name] = mle_values[i]
                i += 1
            else:
                for dim, max_degree in dims[:]:
                    coefficient_name = LinearCoef.coefficient_name(dim, dim_to_coordinate_name)
                    coef_template = LinearCoef.coef_template_str(param_name, coefficient_name)
                    offset = LinearCoef.offset_from_coefficient_name(coefficient_name)
                    for k in range(1, max_degree + 1):
                        coef_name = coef_template.format(k + offset)
                        coef_dict[coef_name] = mle_values[i]
                        i += 1
        # Add potential climatic effects
        if param_name_to_name_of_the_climatic_effects is not None:
            if linear_effects[j] in [False, None]:
                for name in param_name_to_name_of_the_climatic_effects[param_name]:
                    coef_name = param_name + name
                    coef_dict[coef_name] = mle_values[i]
                    i += 1
            if linear_effects[j] in [None, True]:
                for name in param_name_to_name_of_the_climatic_effects[param_name]:
                    coef_name = param_name + name + AbstractCoordinates.COORDINATE_T
                    coef_dict[coef_name] = mle_values[i]
                    i += 1

    if type_for_mle == "Gumbel":
        assert len(coef_dict) == len(mle_values) + 1
    else:
        assert len(coef_dict) == len(mle_values)
    return coef_dict
