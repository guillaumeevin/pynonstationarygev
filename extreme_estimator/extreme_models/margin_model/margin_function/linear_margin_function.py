from typing import Dict, List, Tuple

from extreme_estimator.extreme_models.margin_model.margin_function.independent_margin_function import \
    IndependentMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.margin_model.param_function.param_function import ConstantParamFunction, \
    ParamFunction, LinearParamFunction
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearMarginFunction(IndependentMarginFunction):
    """ Margin Function, where each parameter can augment linearly along any dimension.

        dim = 0 correspond to the intercept
        dim = 1 correspond to the coordinate X
        dim = 2 correspond to the coordinate Y
        dim = 3 correspond to the coordinate Z

        gev_param_name_to_linear_dims             maps each parameter of the GEV distribution to its linear dimensions

        gev_param_name_to_linear_coef             maps each parameter of the GEV distribution to its linear coefficients

        gev_param_name_to_start_fit_linear_coef   maps each parameter of the GEV distribution to its starting fitting
                                                   value for the linear coefficients
    """

    def __init__(self, coordinates: AbstractCoordinates,
                 gev_param_name_to_linear_dims: Dict[str, List[int]],
                 gev_param_name_to_linear_coef: Dict[str, LinearCoef]):
        super().__init__(coordinates)
        self.gev_param_name_to_linear_coef = gev_param_name_to_linear_coef
        self.gev_param_name_to_linear_dims = gev_param_name_to_linear_dims
        # Build gev_parameter_to_param_function dictionary
        self.gev_param_name_to_param_function = {}  # type: Dict[str, ParamFunction]

        # Check the linear_dim are well-defined with respect to the coordinates
        for linear_dims in self.gev_param_name_to_linear_dims.values():
            for dim in linear_dims:
                assert 0 < dim <= coordinates.nb_columns, "dim={}, nb_columns={}".format(dim, coordinates.nb_columns)

        # Map each gev_param_name to its corresponding param_function
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            linear_coef = self.gev_param_name_to_linear_coef[gev_param_name]
            # By default, if linear_dims are not specified, a constantParamFunction is chosen
            if gev_param_name not in self.gev_param_name_to_linear_dims.keys():
                param_function = ConstantParamFunction(constant=linear_coef.get_coef(dim=0))
            # Otherwise, we fit a LinearParamFunction
            else:
                param_function = LinearParamFunction(linear_dims=self.gev_param_name_to_linear_dims[gev_param_name],
                                                     coordinates=self.coordinates.coordinates_values,
                                                     linear_coef=linear_coef)
            # Add the param_function to the dictionary
            self.gev_param_name_to_param_function[gev_param_name] = param_function

    @classmethod
    def from_margin_coeff_dict(cls, coordinates, margin_coeff_dict):
        pass

    @property
    def form_dict(self) -> dict:
        form_dict = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            linear_dims = self.gev_param_name_to_linear_dims.get(gev_param_name, [])
            form_dict.update(self.gev_param_name_to_linear_coef[gev_param_name].form_dict(linear_dims=linear_dims))
        return form_dict

    @property
    def coef_dict(self) -> dict:
        coef_dict = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            linear_dims = self.gev_param_name_to_linear_dims.get(gev_param_name, [])
            coef_dict.update(self.gev_param_name_to_linear_coef[gev_param_name].coef_dict(linear_dims=linear_dims))
        return coef_dict
