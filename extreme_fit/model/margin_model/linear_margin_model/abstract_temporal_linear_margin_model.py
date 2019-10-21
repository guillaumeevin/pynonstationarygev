import numpy as np
import pandas as pd

from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_extremes import ResultFromExtremes
from extreme_fit.model.result_from_model_fit.result_from_ismev import ResultFromIsmev
from extreme_fit.model.utils import r, ro, get_null, get_margin_formula_extremes, get_coord, get_coord_df
from extreme_fit.model.utils import safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalLinearMarginModel(LinearMarginModel):
    """Linearity only with respect to the temporal coordinates"""
    ISMEV_GEV_FIT_METHOD_STR = 'isMev.gev.fit'
    EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR = 'extRemes.fevd.Bayesian'

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None, fit_method=ISMEV_GEV_FIT_METHOD_STR):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample, starting_point)
        assert fit_method in [self.ISMEV_GEV_FIT_METHOD_STR, self.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR]
        self.fit_method = fit_method

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> AbstractResultFromModelFit:
        assert data.shape[1] == len(df_coordinates_temp.values)
        x = ro.FloatVector(data[0])
        if self.fit_method == self.ISMEV_GEV_FIT_METHOD_STR:
            return self.ismev_gev_fit(x, df_coordinates_temp)
        if self.fit_method == self.EXTREMES_FEVD_BAYESIAN_FIT_METHOD_STR:
            return self.extremes_fevd_bayesian_fit(x, df_coordinates_temp)

    # Gev Fit with isMev package

    def ismev_gev_fit(self, x, df_coordinates_temp) -> ResultFromIsmev:
        y = df_coordinates_temp.values
        res = safe_run_r_estimator(function=r('gev.fit'), use_start=self.use_start_value,
                                   xdat=x, y=y, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.margin_function_start_fit.gev_param_name_to_dims)

    # Gev fit with extRemes package

    def extremes_fevd_bayesian_fit(self, x, df_coordinates_temp) -> ResultFromExtremes:
        # Disable the use of log sigma parametrization
        r_type_argument_kwargs = {'use.phi': False,
                                  'verbose': False}
        r_type_argument_kwargs.update(get_margin_formula_extremes(self.margin_function_start_fit.form_dict))
        y = get_coord_df(df_coordinates_temp)
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   method='Bayesian',
                                   priorFun="fevdPriorCustom",
                                   priorParams=r.list(q=r.c(6), p=r.c(9)),
                                   iter=5000,
                                   **r_type_argument_kwargs
                                   )
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
