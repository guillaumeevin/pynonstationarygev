import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.gev_params import GevParams


class LinearMarginModel(AbstractMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_dims=None):
        # Load sample coef
        self.default_params_sample = self.default_param_name_and_dim_to_coef()
        linear_coef_sample = self.gev_param_name_to_linear_coef(param_name_and_dim_to_coef=self.params_sample)
        self.margin_function_sample = LinearMarginFunction(coordinates=self.coordinates,
                                                           gev_param_name_to_linear_coef=linear_coef_sample,
                                                           gev_param_name_to_linear_dims=gev_param_name_to_linear_dims)

        # Load start fit coef
        self.default_params_start_fit = self.default_param_name_and_dim_to_coef()
        linear_coef_start_fit = self.gev_param_name_to_linear_coef(param_name_and_dim_to_coef=self.params_start_fit)
        self.margin_function_start_fit = LinearMarginFunction(coordinates=self.coordinates,
                                                              gev_param_name_to_linear_coef=linear_coef_start_fit,
                                                              gev_param_name_to_linear_dims=gev_param_name_to_linear_dims)

    @staticmethod
    def default_param_name_and_dim_to_coef() -> dict:
        default_intercept = 1
        default_slope = 0.01
        gev_param_name_and_dim_to_coef = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            gev_param_name_and_dim_to_coef[(gev_param_name, 0)] = default_intercept
            for dim in [1, 2, 3]:
                gev_param_name_and_dim_to_coef[(gev_param_name, dim)] = default_slope
        return gev_param_name_and_dim_to_coef

    @staticmethod
    def gev_param_name_to_linear_coef(param_name_and_dim_to_coef):
        gev_param_name_to_linear_coef = {}
        for gev_param_name in GevParams.GEV_PARAM_NAMES:
            dim_to_coef = {dim: param_name_and_dim_to_coef[(gev_param_name, dim)] for dim in [0, 1, 2, 3]}
            linear_coef = LinearCoef(gev_param_name=gev_param_name, dim_to_coef=dim_to_coef)
            gev_param_name_to_linear_coef[gev_param_name] = linear_coef
        return gev_param_name_to_linear_coef

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray,
                                  coordinates_values: np.ndarray) -> AbstractMarginFunction:
        return self.margin_function_start_fit


class ConstantMarginModel(LinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({})


class LinearShapeDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.GEV_SHAPE: [1]})


class LinearScaleDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.GEV_SCALE: [1]})


class LinearShapeDim1and2MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.GEV_SHAPE: [1, 2]})


class LinearAllParametersDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.GEV_SHAPE: [1],
                                       GevParams.GEV_LOC: [1],
                                       GevParams.GEV_SCALE: [1]})


class LinearAllParametersAllDimsMarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        all_dims = list(range(1, self.coordinates.nb_columns + 1))
        super().load_margin_functions({GevParams.GEV_SHAPE: all_dims.copy(),
                                       GevParams.GEV_LOC: all_dims.copy(),
                                       GevParams.GEV_SCALE: all_dims.copy()})
