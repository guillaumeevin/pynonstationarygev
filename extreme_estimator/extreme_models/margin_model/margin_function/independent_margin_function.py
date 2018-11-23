from typing import Dict, List, Tuple

import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.param_function import ConstantParamFunction, \
    LinearOneAxisParamFunction, ParamFunction, LinearParamFunction
from extreme_estimator.gev_params import GevParams
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractSpatialCoordinates


class IndependentMarginFunction(AbstractMarginFunction):
    """Margin Function where each parameter of the GEV are modeled independently"""

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates, default_params: GevParams):
        """Attribute 'gev_param_name_to_param_function' maps each GEV parameter to its corresponding function"""
        super().__init__(spatial_coordinates, default_params)
        self.gev_param_name_to_param_function = None  # type: Dict[str, ParamFunction]

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        """Each GEV parameter is computed independently through its corresponding param_function"""
        assert self.gev_param_name_to_param_function is not None
        assert len(self.gev_param_name_to_param_function) == 3
        gev_params = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            param_function = self.gev_param_name_to_param_function[gev_param_name]
            gev_params[gev_param_name] = param_function.get_gev_param_value(coordinate)
        return GevParams.from_dict(gev_params)


class LinearMarginFunction(IndependentMarginFunction):
    """ Margin Function, where each parameter can augment linearly as follows:
        On the extremal point along all the dimension, the GEV parameters will equal the default_param value
        Then, it will augment linearly along a single linear axis"""

    def __init__(self, spatial_coordinates: AbstractSpatialCoordinates,
                 default_params: GevParams,
                 gev_param_name_to_linear_axes: Dict[str, List[int]],
                 gev_param_name_and_axis_to_start_fit: Dict[Tuple[str, int], float] = None):
        """
        -Attribute 'gev_param_name_to_linear_axis'        maps each GEV parameter to its corresponding function
        -Attribute 'gev_param_name_and_axis_to_start_fit' maps each tuple (GEV parameter, axis) to its start value for
            fitting (by default equal to 1). Also start value for the intercept is equal to 0 by default."""
        super().__init__(spatial_coordinates, default_params)
        self.gev_param_name_and_axis_to_start_fit = gev_param_name_and_axis_to_start_fit
        self.gev_param_name_to_linear_axes = gev_param_name_to_linear_axes

        # Check the axes are well-defined with respect to the coordinates
        for axes in self.gev_param_name_to_linear_axes.values():
            assert all([axis < np.ndim(spatial_coordinates.coordinates) for axis in axes])

        # Build gev_parameter_to_param_function dictionary
        self.gev_param_name_to_param_function = {}  # type: Dict[str, ParamFunction]
        # Map each gev_param_name to its corresponding param_function
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            # By default, if gev_param_name linear_axis is not specified, a constantParamFunction is chosen
            if gev_param_name not in self.gev_param_name_to_linear_axes.keys():
                param_function = ConstantParamFunction(constant=self.default_params[gev_param_name])
            # Otherwise, we fit a LinearParamFunction
            else:
                param_function = LinearParamFunction(linear_axes=self.gev_param_name_to_linear_axes[gev_param_name],
                                                     coordinates=self.spatial_coordinates.coordinates,
                                                     start=self.default_params[gev_param_name])
                # Some check on the Linear param function
                if gev_param_name == GevParams.GEV_SCALE:
                    assert param_function.end > param_function.start, 'Impossible linear rate for Scale parameter'

            # Add the param_function to the dictionary
            self.gev_param_name_to_param_function[gev_param_name] = param_function

    @property
    def fit_marge_form_dict(self) -> dict:
        """
        Example of formula that could be specified:
        loc.form = loc ~ coord_x
        scale.form = scale ~ coord_y
        shape.form = shape ~ coord_x+coord_y
        :return:
        """
        fit_marge_form_dict = {}
        axis_to_name = {i: name for i, name in enumerate(AbstractSpatialCoordinates.COORDINATE_NAMES)}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            axes = self.gev_param_name_to_linear_axes.get(gev_param_name, None)
            formula_str = '1' if axes is None else '+'.join([axis_to_name[axis] for axis in axes])
            fit_marge_form_dict[gev_param_name + '.form'] = gev_param_name + ' ~ ' + formula_str
        return fit_marge_form_dict

    @property
    def margin_start_dict(self) -> dict:
        # Define default values
        default_start_fit_coef = 1.0
        default_start_fit_intercept = 0.0
        # Build the dictionary containing all the parameters
        margin_start_dict = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            coef_template_str = gev_param_name + 'Coeff{}'
            # Constant param must be specified for all the parameters
            margin_start_dict[coef_template_str.format(1)] = default_start_fit_intercept
            for j, axis in enumerate(self.gev_param_name_to_linear_axes.get(gev_param_name, []), 2):
                if self.gev_param_name_and_axis_to_start_fit is None:
                    coef = default_start_fit_coef
                else:
                    coef = self.gev_param_name_and_axis_to_start_fit.get((gev_param_name, axis), default_start_fit_coef)
                margin_start_dict[coef_template_str.format(j)] = coef
        return margin_start_dict
