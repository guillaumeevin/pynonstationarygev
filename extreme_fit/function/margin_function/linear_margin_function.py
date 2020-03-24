from typing import Dict, List, Union

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.function.margin_function.parametric_margin_function import \
    ParametricMarginFunction
from extreme_fit.function.param_function.abstract_coef import AbstractCoef
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.function.param_function.param_function import AbstractParamFunction, \
    LinearParamFunction
from extreme_fit.distribution.gev.gev_params import GevParams
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
                 gev_param_name_to_coef: Dict[str, AbstractCoef], starting_point: Union[None, int] = None,
                 params_class: type = GevParams):
        self.gev_param_name_to_coef = None  # type: Union[None, Dict[str, LinearCoef]]
        super().__init__(coordinates, gev_param_name_to_dims, gev_param_name_to_coef, starting_point,
                         params_class)

    def load_specific_param_function(self, gev_param_name) -> AbstractParamFunction:
        return LinearParamFunction(dims=self.gev_param_name_to_dims[gev_param_name],
                                   coordinates=self.coordinates.coordinates_values(),
                                   linear_coef=self.gev_param_name_to_coef[gev_param_name])

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
    def coef_dict(self) -> Dict[str, float]:
        coef_dict = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            dims = self.gev_param_name_to_dims.get(gev_param_name, [])
            coef = self.gev_param_name_to_coef[gev_param_name]
            coef_dict.update(coef.coef_dict(dims, self.idx_to_coefficient_name(self.coordinates)))
        return coef_dict

    @property
    def form_dict(self) -> Dict[str, str]:
        form_dict = {}
        for gev_param_name in self.params_class.PARAM_NAMES:
            linear_dims = self.gev_param_name_to_dims.get(gev_param_name, [])
            # Load spatial form_dict (only if we have some spatial coordinates)
            if self.coordinates.has_spatial_coordinates:
                spatial_names = [name for name in self.coordinates.spatial_coordinates_names
                                 if self.coefficient_name_to_dim(self.coordinates)[name] in linear_dims]
                spatial_form = self.gev_param_name_to_coef[gev_param_name].spatial_form_dict(spatial_names)
                form_dict.update(spatial_form)
            # Load temporal form dict (only if we have some temporal coordinates)
            if self.coordinates.has_temporal_coordinates:
                temporal_names = [name for name in self.coordinates.temporal_coordinates_names
                                  if self.coefficient_name_to_dim(self.coordinates)[name] in linear_dims]
                temporal_form = self.gev_param_name_to_coef[gev_param_name].temporal_form_dict(temporal_names)
                # Specifying a formula '~ 1' creates a bug in fitspatgev of SpatialExtreme R package
                assert not any(['1' in formula for formula in temporal_form.values()])
                form_dict.update(temporal_form)
        return form_dict

    # Properties for the location parameter

    def get_coef(self, gev_param_name, coef_name):
        idx = 1 if coef_name in [AbstractCoordinates.COORDINATE_T, LinearCoef.INTERCEPT_NAME] \
            else AbstractCoordinates.COORDINATE_SPATIAL_NAMES.index(coef_name) + 2
        return self.coef_dict[LinearCoef.coef_template_str(gev_param_name, coef_name).format(idx)]
    
    @property
    def mu1_temporal_trend(self):
        return self.coef_dict[LinearCoef.coef_template_str(AbstractParams.LOC, AbstractCoordinates.COORDINATE_T).format(1)]

    @property
    def mu_intercept(self):
        return self.coef_dict[LinearCoef.coef_template_str(AbstractParams.LOC, LinearCoef.INTERCEPT_NAME).format(1)]

    @property
    def mu_longitude_trend(self):
        return self.coef_dict[LinearCoef.coef_template_str(AbstractParams.LOC, AbstractCoordinates.COORDINATE_X).format(2)]

    @property
    def mu_latitude_trend(self):
        return self.coef_dict[LinearCoef.coef_template_str(AbstractParams.LOC, AbstractCoordinates.COORDINATE_Y).format(3)]