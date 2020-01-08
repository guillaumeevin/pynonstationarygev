from enum import Enum

import numpy as np
import pandas as pd

from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import AbstractResultFromExtremes, ResultFromBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_mle_extremes import ResultFromMleExtremes
from extreme_fit.model.result_from_model_fit.result_from_ismev import ResultFromIsmev
from extreme_fit.model.utils import r, ro, get_null, get_margin_formula_extremes, get_coord_df
from extreme_fit.model.utils import safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class TemporalMarginFitMethod(Enum):
    is_mev_gev_fit = 0
    extremes_fevd_bayesian = 1
    extremes_fevd_mle = 2
    extremes_fevd_gmle = 3


class AbstractTemporalLinearMarginModel(LinearMarginModel):
    """Linearity only with respect to the temporal coordinates"""

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False, params_start_fit=None,
                 params_sample=None, starting_point=None,
                 fit_method=TemporalMarginFitMethod.is_mev_gev_fit,
                 nb_iterations_for_bayesian_fit=5000,
                 params_start_fit_bayesian=None,
                 type_for_MLE="GEV"):
        super().__init__(coordinates, use_start_value, params_start_fit, params_sample, starting_point)
        self.type_for_mle = type_for_MLE
        self.params_start_fit_bayesian = params_start_fit_bayesian
        self.nb_iterations_for_bayesian_fit = nb_iterations_for_bayesian_fit
        assert isinstance(fit_method, TemporalMarginFitMethod), fit_method
        self.fit_method = fit_method

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> AbstractResultFromModelFit:
        assert data.shape[1] == len(df_coordinates_temp.values)
        x = ro.FloatVector(data[0])
        if self.fit_method == TemporalMarginFitMethod.is_mev_gev_fit:
            return self.ismev_gev_fit(x, df_coordinates_temp)
        if self.fit_method == TemporalMarginFitMethod.extremes_fevd_bayesian:
            return self.extremes_fevd_bayesian_fit(x, df_coordinates_temp)
        if self.fit_method in [TemporalMarginFitMethod.extremes_fevd_mle, TemporalMarginFitMethod.extremes_fevd_gmle]:
            return self.extremes_fevd_mle_related_fit(x, df_coordinates_temp)

    # Gev Fit with isMev package

    def ismev_gev_fit(self, x, df_coordinates_temp) -> ResultFromIsmev:
        y = df_coordinates_temp.values
        res = safe_run_r_estimator(function=r('gev.fit'), use_start=self.use_start_value,
                                   xdat=x, y=y, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.margin_function_start_fit.gev_param_name_to_dims)

    # Gev fit with extRemes package

    def extremes_fevd_mle_related_fit(self, x, df_coordinates_temp) -> AbstractResultFromExtremes:
        r_type_argument_kwargs, y = self.extreme_arguments(df_coordinates_temp)
        if self.fit_method == TemporalMarginFitMethod.extremes_fevd_mle:
            method = "MLE"
        elif self.fit_method == TemporalMarginFitMethod.extremes_fevd_gmle:
            method = "GMLE"
        else:
            raise ValueError('wrong method')
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   type=self.type_for_mle,
                                   method=method,
                                   **r_type_argument_kwargs
                                   )
        return ResultFromMleExtremes(res, self.margin_function_start_fit.gev_param_name_to_dims,
                                     type_for_mle=self.type_for_mle)

    def extremes_fevd_bayesian_fit(self, x, df_coordinates_temp) -> AbstractResultFromExtremes:
        r_type_argument_kwargs, y = self.extreme_arguments(df_coordinates_temp)
        params_start_fit = self.params_start_fit_bayesian if self.params_start_fit_bayesian is not None else {}
        r_type_argument_kwargs['initial'] = r.list(**params_start_fit)
        # Assert for any non-stationary model that the shape parameter is constant
        # (because the prior function considers that the last parameter should be the shape)
        assert GevParams.SHAPE not in self.margin_function_start_fit.gev_param_name_to_dims \
               or len(self.margin_function_start_fit.gev_param_name_to_dims[GevParams.SHAPE]) == 1
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   method='Bayesian',
                                   priorFun="fevdPriorCustom",
                                   priorParams=r.list(q=r.c(6), p=r.c(9)),
                                   iter=self.nb_iterations_for_bayesian_fit,
                                   **r_type_argument_kwargs
                                   )
        return ResultFromBayesianExtremes(res, self.margin_function_start_fit.gev_param_name_to_dims)

    def extreme_arguments(self, df_coordinates_temp):
        # Disable the use of log sigma parametrization
        r_type_argument_kwargs = {'use.phi': False,
                                  'verbose': False}
        # Load parameters
        r_type_argument_kwargs.update(get_margin_formula_extremes(self.margin_function_start_fit.form_dict))
        y = get_coord_df(df_coordinates_temp)
        return r_type_argument_kwargs, y

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
