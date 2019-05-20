import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit, ResultFromIsmev
from extreme_estimator.extreme_models.utils import r, ro, get_null
from extreme_estimator.extreme_models.utils import safe_run_r_estimator
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class TemporalLinearMarginModel(LinearMarginModel):
    # Linearity only with respect to the temporal coordinates

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample, starting_point)

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> ResultFromFit:
        # Modify df_coordinates_temp
        df_coordinates_temp = self.add_starting_temporal_point(df_coordinates_temp)
        # Gev Fit
        assert data.shape[1] == len(df_coordinates_temp.values)
        res = safe_run_r_estimator(function=r('gev.fit'), use_start=self.use_start_value,
                                   xdat=ro.FloatVector(data[0]), y=df_coordinates_temp.values, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.margin_function_start_fit.gev_param_name_to_dims)

    @property
    def mul(self):
        return get_null()

    @property
    def sigl(self):
        return get_null()

    @property
    def shl(self):
        return get_null()


class StationaryStationModel(TemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({})


class NonStationaryLocationStationModel(TemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.LOC: [self.coordinates.idx_temporal_coordinates]})

    @property
    def mul(self):
        return 1


class NonStationaryScaleStationModel(TemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SCALE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def sigl(self):
        return 1


class NonStationaryShapeStationModel(TemporalLinearMarginModel):

    def load_margin_functions(self, gev_param_name_to_dims=None):
        super().load_margin_functions({GevParams.SHAPE: [self.coordinates.idx_temporal_coordinates]})

    @property
    def shl(self):
        return 1
