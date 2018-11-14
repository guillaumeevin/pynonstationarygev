import numpy as np

from extreme_estimator.R_model.margin_model.abstract_margin_function import ConstantMarginFunction, \
    AbstractMarginFunction
from extreme_estimator.R_model.margin_model.abstract_margin_model import AbstractMarginModel
from extreme_estimator.R_model.gev.gev_parameters import GevParams
from spatio_temporal_dataset.spatial_coordinates.abstract_spatial_coordinates import AbstractSpatialCoordinates


class SmoothMarginModel(AbstractMarginModel):
    pass


class ConstantMarginModel(SmoothMarginModel):

    def load_margin_functions(self, margin_function_class: type = None):
        self.default_params_sample = GevParams(1.0, 1.0, 1.0).to_dict()
        self.default_params_start_fit = GevParams(1.0, 1.0, 1.0).to_dict()
        super().load_margin_functions(margin_function_class=ConstantMarginFunction)

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, coordinates: np.ndarray) -> AbstractMarginFunction:
        return self.margin_function_start_fit


class LinearShapeMarginModel(SmoothMarginModel):
    pass


if __name__ == '__main__':
    pass
