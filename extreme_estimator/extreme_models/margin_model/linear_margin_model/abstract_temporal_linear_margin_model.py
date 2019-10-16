import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_estimator.extreme_models.result_from_fit import ResultFromFit, ResultFromIsmev, ResultFromExtremes
from extreme_estimator.extreme_models.utils import r, ro, get_null
from extreme_estimator.extreme_models.utils import safe_run_r_estimator
from extreme_estimator.margin_fits.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalLinearMarginModel(LinearMarginModel):
    """Linearity only with respect to the temporal coordinates"""
    ISMEV_GEV_FIT_METHOD_STR = 'isMev.gev.fit'
    EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR = 'extRemes.fevd.Bayesian'

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None, fit_method='isMev.gev.fit'):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample, starting_point)
        assert fit_method in [self.ISMEV_GEV_FIT_METHOD_STR, self.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR]
        self.fit_method = fit_method

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> ResultFromFit:
        assert data.shape[1] == len(df_coordinates_temp.values)
        if self.fit_method == self.ISMEV_GEV_FIT_METHOD_STR:
            return self.ismev_gev_fit(data, df_coordinates_temp)
        if self.fit_method == self.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR:
            return self.extremes_fevd_bayesian_fit(data, df_coordinates_temp)

    # Gev Fit with isMev package

    def ismev_gev_fit(self, data: np.ndarray, df_coordinates_temp: pd.DataFrame) -> ResultFromIsmev:
        res = safe_run_r_estimator(function=r('gev.fit'), use_start=self.use_start_value,
                                   xdat=ro.FloatVector(data[0]), y=df_coordinates_temp.values, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.margin_function_start_fit.gev_param_name_to_dims)

    # Gev fit with extRemes package

    def extremes_fevd_bayesian_fit(self, data, df_coordinates_temp) -> ResultFromExtremes:
        res = safe_run_r_estimator(function=r('fevd_fixed'), use_start=self.use_start_value,
                                   xdat=ro.FloatVector(data[0]), y=df_coordinates_temp.values, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromExtremes(res, self.margin_function_start_fit.gev_param_name_to_dims)

    # Default arguments for all methods

    @property
    def mul(self):
        return get_null()

    @property
    def sigl(self):
        return get_null()

    @property
    def shl(self):
        return get_null()

    @property
    def siglink(self):
        return r('identity')





