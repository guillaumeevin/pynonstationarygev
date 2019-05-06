from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.margin_model.parametric_margin_model import ParametricMarginModel
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class LinearMarginModel(ParametricMarginModel):

    @classmethod
    def from_coef_list(cls, coordinates, gev_param_name_to_coef_list):
        params = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            for idx, coef in enumerate(gev_param_name_to_coef_list[gev_param_name], -1):
                params[(gev_param_name, idx)] = coef
        return cls(coordinates, params_sample=params, params_start_fit=params)

    def load_margin_functions(self, gev_param_name_to_dims=None):
        assert gev_param_name_to_dims is not None, 'LinearMarginModel cannot be used for sampling/fitting \n' \
                                                   'load_margin_functions needs to be implemented in child class'
        # Load default params (with a dictionary format to enable quick replacement)
        # IMPORTANT: Using a dictionary format enable using the default/user params methodology
        self.default_params_sample = self.default_param_name_and_dim_to_coef
        self.default_params_start_fit = self.default_param_name_and_dim_to_coef

        # Load sample coef
        coef_sample = self.gev_param_name_to_linear_coef(param_name_and_dim_to_coef=self.params_sample)
        self.margin_function_sample = LinearMarginFunction(coordinates=self.coordinates,
                                                           gev_param_name_to_coef=coef_sample,
                                                           gev_param_name_to_dims=gev_param_name_to_dims,
                                                           starting_point=self.starting_point)

        # Load start fit coef
        coef_start_fit = self.gev_param_name_to_linear_coef(param_name_and_dim_to_coef=self.params_start_fit)
        self.margin_function_start_fit = LinearMarginFunction(coordinates=self.coordinates,
                                                              gev_param_name_to_coef=coef_start_fit,
                                                              gev_param_name_to_dims=gev_param_name_to_dims,
                                                              starting_point=self.starting_point)

    @property
    def default_param_name_and_dim_to_coef(self) -> dict:
        default_intercept = 1
        default_slope = 0.01
        gev_param_name_and_dim_to_coef = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            gev_param_name_and_dim_to_coef[(gev_param_name, -1)] = default_intercept
            for dim in self.coordinates.coordinates_dims:
                gev_param_name_and_dim_to_coef[(gev_param_name, dim)] = default_slope
        return gev_param_name_and_dim_to_coef

    def gev_param_name_to_linear_coef(self, param_name_and_dim_to_coef):
        gev_param_name_to_linear_coef = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            idx_to_coef = {idx: param_name_and_dim_to_coef[(gev_param_name, idx)] for idx in
                           [-1] + self.coordinates.coordinates_dims}
            linear_coef = LinearCoef(gev_param_name=gev_param_name, idx_to_coef=idx_to_coef)
            gev_param_name_to_linear_coef[gev_param_name] = linear_coef
        return gev_param_name_to_linear_coef


class ConstantMarginModel(LinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({})


class LinearShapeDim0MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0]})


class LinearScaleDim0MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [0]})


class LinearShapeDim0and1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0, 1]})


class LinearAllParametersDim0MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0],
                                       GevParams.LOC: [0],
                                       GevParams.SCALE: [0]})


class LinearMarginModelExample(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0],
                                       GevParams.LOC: [1],
                                       GevParams.SCALE: [0]})


class LinearAllParametersAllDimsMarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: self.coordinates.coordinates_dims,
                                       GevParams.LOC: self.coordinates.coordinates_dims,
                                       GevParams.SCALE: self.coordinates.coordinates_dims})


class LinearAllParametersTwoFirstCoordinatesMarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0, 1],
                                       GevParams.LOC: [0, 1],
                                       GevParams.SCALE: [0, 1]})


class LinearAllTwoStatialCoordinatesLocationLinearMarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [0, 1],
                                       GevParams.LOC: [0, 1, 2],
                                       GevParams.SCALE: [0, 1]})


# Some renaming that defines the stationary and non-stationary models of reference
class LinearStationaryMarginModel(LinearAllParametersTwoFirstCoordinatesMarginModel):
    pass


class LinearNonStationaryMarginModel(LinearAllTwoStatialCoordinatesLocationLinearMarginModel):
    pass
