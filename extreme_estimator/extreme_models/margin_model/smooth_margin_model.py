import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.result_from_fit import ResultFromFit
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.extreme_models.margin_model.margin_function.linear_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.param_function.linear_coef import LinearCoef
from extreme_estimator.extreme_models.utils import safe_run_r_estimator, r, get_coord, \
    get_margin_formula
from extreme_estimator.margin_fits.gev.gev_params import GevParams


class LinearMarginModel(AbstractMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_dims=None):
        assert gev_param_name_to_linear_dims is not None, 'LinearMarginModel cannot be used for sampling/fitting \n' \
                                                          'load_margin_functions needs to be implemented in child class'
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
        for gev_param_name in GevParams.PARAM_NAMES:
            gev_param_name_and_dim_to_coef[(gev_param_name, 0)] = default_intercept
            for dim in [1, 2, 3]:
                gev_param_name_and_dim_to_coef[(gev_param_name, dim)] = default_slope
        return gev_param_name_and_dim_to_coef

    @staticmethod
    def gev_param_name_to_linear_coef(param_name_and_dim_to_coef):
        gev_param_name_to_linear_coef = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            dim_to_coef = {dim: param_name_and_dim_to_coef[(gev_param_name, dim)] for dim in [0, 1, 2, 3]}
            linear_coef = LinearCoef(gev_param_name=gev_param_name, dim_to_coef=dim_to_coef)
            gev_param_name_to_linear_coef[gev_param_name] = linear_coef
        return gev_param_name_to_linear_coef

    @classmethod
    def from_coef_list(cls, coordinates, gev_param_name_to_coef_list):
        params = {}
        for gev_param_name in GevParams.PARAM_NAMES:
            for dim, coef in enumerate(gev_param_name_to_coef_list[gev_param_name]):
                params[(gev_param_name, dim)] = coef
        return cls(coordinates, params_sample=params, params_start_fit=params)

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, df_coordinates_spatial: pd.DataFrame,
                                  df_coordinates_temporal: pd.DataFrame) -> ResultFromFit:
        # The reshaping on the line below is only valid if we have a single observation per spatio-temporal point
        if maxima_gev.shape[1] == 1:
            maxima_gev = maxima_gev.reshape([len(df_coordinates_temporal), len(df_coordinates_spatial)])
        data = np.transpose(maxima_gev)

        fit_params = get_margin_formula(self.margin_function_start_fit.form_dict)

        # Covariables
        covariables = get_coord(df_coordinates=df_coordinates_spatial)
        fit_params['temp.cov'] = get_coord(df_coordinates=df_coordinates_temporal)

        # Start parameters
        coef_dict = self.margin_function_start_fit.coef_dict
        fit_params['start'] = r.list(**coef_dict)

        return safe_run_r_estimator(function=r.fitspatgev, use_start=self.use_start_value, data=data,
                                    covariables=covariables, **fit_params)


class ConstantMarginModel(LinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({})


class LinearShapeDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [1]})


class LinearScaleDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.SCALE: [1]})


class LinearShapeDim1and2MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [1, 2]})


class LinearAllParametersDim1MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [1],
                                       GevParams.LOC: [1],
                                       GevParams.SCALE: [1]})


class LinearMarginModelExample(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [1],
                                       GevParams.LOC: [2],
                                       GevParams.SCALE: [1]})


class LinearAllParametersAllDimsMarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_dims=None):
        all_dims = list(range(1, self.coordinates.nb_coordinates + 1))
        super().load_margin_functions({GevParams.SHAPE: all_dims.copy(),
                                       GevParams.LOC: all_dims.copy(),
                                       GevParams.SCALE: all_dims.copy()})
