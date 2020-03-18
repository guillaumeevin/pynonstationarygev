from typing import Dict, List, Union

import numpy as np

from extreme_fit.function.margin_function.independent_margin_function import \
    IndependentMarginFunction
from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.param_function import AbstractParamFunction, \
    ConstantParamFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class ParametricMarginFunction(IndependentMarginFunction):
    """
    ParametricMarginFunction each parameter of the GEV will:

        -depend on some integer dimensions (dimension 1 or/and dimension 2 for instance).
        Coordinate name corresponding to the dimension depends on the order of the columns of self.coordinates
        gev_param_name_to_dims maps each GEV parameter to its corresponding dimensions

        -have a set of all potential coefficient that could be used to define a function
        gev_param_name_to_coef maps each GEV parameter to an AbstractCoef object. This object contains

        -combining the integer dimensions & the set of all potential coefficient
        to keep only the relevant coefficient, and build the corresponding function from that
        gev_param_name_to_param_function maps each GEV parameter to a AbstractParamFunction object.

    """

    COEF_CLASS = None

    def __init__(self, coordinates: AbstractCoordinates, gev_param_name_to_dims: Dict[str, List[int]],
                 gev_param_name_to_coef: Dict[str, AbstractCoef], starting_point: Union[None, int] = None):
        # Starting point for the trend is the same for all the parameters
        self.starting_point = starting_point
        super().__init__(coordinates)
        self.gev_param_name_to_dims = gev_param_name_to_dims  # type: Dict[str, List[int]]

        # Check the dimension are well-defined with respect to the coordinates
        for dims in self.gev_param_name_to_dims.values():
            for dim in dims:
                assert 0 <= dim < coordinates.nb_coordinates, \
                    "dim={}, nb_columns={}".format(dim, coordinates.nb_coordinates)

        self.gev_param_name_to_coef = gev_param_name_to_coef  # type: Dict[str, AbstractCoef]

        # Build gev_parameter_to_param_function dictionary
        self.gev_param_name_to_param_function = {}  # type: Dict[str, AbstractParamFunction]
        # Map each gev_param_name to its corresponding param_function
        for gev_param_name in GevParams.PARAM_NAMES:
            # By default, if dims are not specified, a constantParamFunction is chosen
            if self.gev_param_name_to_dims.get(gev_param_name) is None:
                param_function = ConstantParamFunction(constant=self.gev_param_name_to_coef[gev_param_name].intercept)
            # Otherwise, we load a specific param function
            else:
                param_function = self.load_specific_param_function(gev_param_name)
            # In both cases, we add the param_function to the dictionary
            self.gev_param_name_to_param_function[gev_param_name] = param_function

    def load_specific_param_function(self, gev_param_name) -> AbstractParamFunction:
        raise NotImplementedError

    @property
    def transformed_starting_point(self):
        return self.coordinates.temporal_coordinates.transformation.transform_array(np.array([self.starting_point]))

    def get_gev_params(self, coordinate: np.ndarray, is_transformed: bool = True) -> GevParams:
        if self.starting_point is not None:
            starting_point = self.transformed_starting_point if is_transformed else self.starting_point
            # Shift temporal coordinate to enable to model temporal trend with starting point
            assert self.coordinates.has_temporal_coordinates
            assert 0 <= self.coordinates.idx_temporal_coordinates < len(coordinate)
            if coordinate[self.coordinates.idx_temporal_coordinates] < starting_point:
                coordinate[self.coordinates.idx_temporal_coordinates] = starting_point
        return super().get_gev_params(coordinate, is_transformed=is_transformed)

    @classmethod
    def from_coef_dict(cls, coordinates: AbstractCoordinates, gev_param_name_to_dims: Dict[str, List[int]],
                       coef_dict: Dict[str, float], starting_point: Union[None, int] = None):
        assert cls.COEF_CLASS is not None, 'a COEF_CLASS class attributes needs to be defined'
        gev_param_name_to_coef = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            dims = gev_param_name_to_dims.get(gev_param_name, [])
            coef = cls.COEF_CLASS.from_coef_dict(coef_dict=coef_dict, gev_param_name=gev_param_name, dims=dims,
                                                 coordinates=coordinates)
            gev_param_name_to_coef[gev_param_name] = coef
        return cls(coordinates, gev_param_name_to_dims, gev_param_name_to_coef, starting_point)

    @property
    def form_dict(self) -> Dict[str, str]:
        raise NotImplementedError
