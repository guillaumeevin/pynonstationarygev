import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_estimator.extreme_models.margin_model.margin_function.independent_margin_function import LinearMarginFunction
from extreme_estimator.extreme_models.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.gev_params import GevParams


class LinearMarginModel(AbstractMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_axis=None):
        self.default_params_sample = GevParams(1.0, 1.0, 1.0).to_dict()
        self.default_params_start_fit = GevParams(1.0, 1.0, 1.0).to_dict()
        self.margin_function_sample = LinearMarginFunction(spatial_coordinates=self.spatial_coordinates,
                                                           default_params=GevParams.from_dict(self.params_sample),
                                                           gev_param_name_to_linear_axis=gev_param_name_to_linear_axis)
        self.margin_function_start_fit = LinearMarginFunction(spatial_coordinates=self.spatial_coordinates,
                                                              default_params=GevParams.from_dict(self.params_start_fit),
                                                              gev_param_name_to_linear_axis=gev_param_name_to_linear_axis)

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, coordinates: np.ndarray) -> AbstractMarginFunction:
        return self.margin_function_start_fit


class ConstantMarginModel(LinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_linear_axis=None):
        super().load_margin_functions({})

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, coordinates: np.ndarray) -> AbstractMarginFunction:
        return self.margin_function_start_fit


class LinearShapeAxis0MarginModel(LinearMarginModel):

    def load_margin_functions(self, margin_function_class: type = None, gev_param_name_to_linear_axis=None):
        super().load_margin_functions({GevParams.GEV_SHAPE: 0})

    # def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, coordinates: np.ndarray) -> AbstractMarginFunction:
    #     pass


if __name__ == '__main__':
    pass
