from typing import Dict, List

import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.abstract_coef import AbstractCoef
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.margin_model.param_function.param_function import ConstantParamFunction, \
    AbstractParamFunction, LinearParamFunction
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearMarginFunction(ParametricMarginFunction):
    """ Margin Function, where each parameter can augment linearly along any dimension.

        dim = 0 correspond to the intercept
        dim = 1 correspond to the first coordinate
        dim = 2 correspond to the second coordinate
        dim = 3 correspond to the third coordinate....

        gev_param_name_to_linear_dims             maps each parameter of the GEV distribution to its linear dimensions

        gev_param_name_to_linear_coef             maps each parameter of the GEV distribution to its linear coefficients

    """

    COEF_CLASS = LinearCoef

    def __init__(self, coordinates: AbstractCoordinates, gev_param_name_to_dims: Dict[str, List[int]],
                 gev_param_name_to_coef: Dict[str, AbstractCoef],
                 starting_point=None):
        # Starting point for the trend is the same for all the parameters
        self.starting_point = starting_point
        self.gev_param_name_to_coef = None  # type: Dict[str, LinearCoef]
        super().__init__(coordinates, gev_param_name_to_dims, gev_param_name_to_coef)

    # @classmethod
    # def from_coef_dict(cls, coordinates: AbstractCoordinates, gev_param_name_to_dims: Dict[str, List[int]],
    #                    coef_dict: Dict[str, float]):
    #     return super().from_coef_dict(coordinates, gev_param_name_to_dims, coef_dict)

    def load_specific_param_function(self, gev_param_name) -> AbstractParamFunction:
        return LinearParamFunction(dims=self.gev_param_name_to_dims[gev_param_name],
                                   coordinates=self.coordinates.coordinates_values(),
                                   linear_coef=self.gev_param_name_to_coef[gev_param_name])

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        if self.starting_point is not None:
            # Shift temporal coordinate to enable to model temporal trend with starting point
            assert self.coordinates.has_temporal_coordinates
            assert 0 <= self.coordinates.idx_temporal_coordinates < len(coordinate)
            if coordinate[self.coordinates.idx_temporal_coordinates] < self.starting_point:
                coordinate[self.coordinates.idx_temporal_coordinates] = self.starting_point
        return super().get_gev_params(coordinate)

    @classmethod
    def idx_to_coefficient_name(cls, coordinates: AbstractCoordinates) -> Dict[int, str]:
        # Intercept correspond to the dimension 0
        idx_to_coefficient_name = {-1: LinearCoef.INTERCEPT_NAME}
        # Coordinates correspond to the dimension starting from 1
        for idx, coordinate_name in enumerate(coordinates.coordinates_names):
            idx_to_coefficient_name[idx] = coordinate_name
        return idx_to_coefficient_name

    @classmethod
    def coefficient_name_to_dim(cls, coordinates: AbstractCoordinates) -> Dict[int, str]:
        return {v: k for k, v in cls.idx_to_coefficient_name(coordinates).items()}

    @property
    def form_dict(self) -> Dict[str, str]:
        form_dict = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            linear_dims = self.gev_param_name_to_dims.get(gev_param_name, [])
            # Load spatial form_dict (only if we have some spatial coordinates)
            if self.coordinates.coordinates_spatial_names:
                spatial_names = [name for name in self.coordinates.coordinates_spatial_names
                                 if self.coefficient_name_to_dim(self.coordinates)[name] in linear_dims]
                spatial_form = self.gev_param_name_to_coef[gev_param_name].spatial_form_dict(spatial_names)
                form_dict.update(spatial_form)
            # Load temporal form dict (only if we have some temporal coordinates)
            if self.coordinates.coordinates_temporal_names:
                temporal_names = [name for name in self.coordinates.coordinates_temporal_names
                                  if self.coefficient_name_to_dim(self.coordinates)[name] in linear_dims]
                temporal_form = self.gev_param_name_to_coef[gev_param_name].temporal_form_dict(temporal_names)
                # Specifying a formula '~ 1' creates a bug in fitspatgev of SpatialExtreme R package
                assert not any(['1' in formula for formula in temporal_form.values()])
                form_dict.update(temporal_form)
        return form_dict
