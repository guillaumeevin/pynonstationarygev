from extreme_fit.function.margin_function.linear_margin_function import LinearMarginFunction
from extreme_fit.function.param_function.linear_coef import LinearCoef
from extreme_fit.model.margin_model.parametric_margin_model import ParametricMarginModel
from extreme_fit.distribution.gev.gev_params import GevParams


class LinearMarginModel(ParametricMarginModel):

    @classmethod
    def from_coef_list(cls, coordinates, param_name_to_coef_list, params_class=GevParams, **kwargs):
        params = {}
        for param_name, coef_list in param_name_to_coef_list.items():
            for idx, coef in enumerate(coef_list, -1):
                params[(param_name, idx)] = coef
        return cls(coordinates, params_user=params, params_class=params_class, **kwargs)

    def load_margin_function(self, param_name_to_dims=None):
        assert param_name_to_dims is not None, 'LinearMarginModel cannot be used for sampling/fitting \n' \
                                               'load_margin_functions needs to be implemented in child class'
        param_name_to_coef = self.param_name_to_linear_coef(param_name_and_dim_to_coef=self.params_sample)
        return LinearMarginFunction(coordinates=self.coordinates,
                                    param_name_to_coef=param_name_to_coef,
                                    param_name_to_dims=param_name_to_dims,
                                    starting_point=self.starting_point,
                                    params_class=self.params_class)

    @property
    def default_params(self) -> dict:
        default_intercept = 1
        default_slope = 0.01
        param_name_and_dim_to_coef = {}
        for param_name in self.params_class.PARAM_NAMES:
            param_name_and_dim_to_coef[(param_name, -1)] = default_intercept
            for dim in self.coordinates.coordinates_dims:
                param_name_and_dim_to_coef[(param_name, dim)] = default_slope
        return param_name_and_dim_to_coef

    def param_name_to_linear_coef(self, param_name_and_dim_to_coef):
        param_name_to_linear_coef = {}
        param_names = list(set([e[0] for e in param_name_and_dim_to_coef.keys()]))
        for param_name in param_names:
            idx_to_coef = {idx: param_name_and_dim_to_coef[(param_name, idx)] for idx in
                           [-1] + self.coordinates.coordinates_dims}
            linear_coef = LinearCoef(param_name=param_name, idx_to_coef=idx_to_coef)
            param_name_to_linear_coef[param_name] = linear_coef
        return param_name_to_linear_coef


class ConstantMarginModel(LinearMarginModel):

    def load_margin_function(self, param_name_to_dims=None):
        return super().load_margin_function({})


class LinearShapeDim0MarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [0]})


class LinearScaleDim0MarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SCALE: [0]})


class LinearShapeDim0and1MarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [0, 1]})


class LinearAllParametersDim0MarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [0],
                                             GevParams.LOC: [0],
                                             GevParams.SCALE: [0]})


class LinearMarginModelExample(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: [0],
                                             GevParams.LOC: [1],
                                             GevParams.SCALE: [0]})


class LinearLocationAllDimsMarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.LOC: self.coordinates.coordinates_dims})


class LinearShapeAllDimsMarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: self.coordinates.coordinates_dims})


class LinearAllParametersAllDimsMarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: self.coordinates.coordinates_dims,
                                             GevParams.LOC: self.coordinates.coordinates_dims,
                                             GevParams.SCALE: self.coordinates.coordinates_dims})


class LinearStationaryMarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: self.coordinates.spatial_coordinates_dims,
                                             GevParams.LOC: self.coordinates.spatial_coordinates_dims,
                                             GevParams.SCALE: self.coordinates.spatial_coordinates_dims})


class LinearNonStationaryLocationMarginModel(LinearMarginModel):

    def load_margin_function(self, margin_function_class: type = None, param_name_to_dims=None):
        return super().load_margin_function({GevParams.SHAPE: self.coordinates.spatial_coordinates_dims,
                                             GevParams.LOC: self.coordinates.coordinates_dims,
                                             GevParams.SCALE: self.coordinates.spatial_coordinates_dims})
