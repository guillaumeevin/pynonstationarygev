import numpy as np
import pandas as pd

from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod, fitmethod_to_str, FEVD_MARGIN_FIT_METHODS, \
    FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    AbstractResultFromExtremes, ResultFromBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_mle_extremes import ResultFromMleExtremes
from extreme_fit.model.result_from_model_fit.result_from_ismev import ResultFromIsmev
from extreme_fit.model.utils import r, ro, get_null, get_margin_formula_extremes, get_coord_df
from extreme_fit.model.utils import safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalLinearMarginModel(LinearMarginModel):
    """Linearity only with respect to the temporal coordinates"""

    def __init__(self, coordinates: AbstractCoordinates,
                 params_sample=None, starting_point=None,
                 fit_method=MarginFitMethod.is_mev_gev_fit,
                 nb_iterations_for_bayesian_fit=5000,
                 params_initial_fit_bayesian=None,
                 type_for_MLE="GEV",
                 params_class=GevParams):
        super().__init__(coordinates, params_sample, starting_point,
                         params_class)
        self.type_for_mle = type_for_MLE
        self.params_initial_fit_bayesian = params_initial_fit_bayesian
        self.nb_iterations_for_bayesian_fit = nb_iterations_for_bayesian_fit
        assert isinstance(fit_method, MarginFitMethod), fit_method
        self.fit_method = fit_method

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> AbstractResultFromModelFit:
        data = data[0]
        assert len(data) == len(df_coordinates_temp.values), 'len(data)={} != len(temp)={}'.format(len(data),
                                                                                                   len(
                                                                                                       df_coordinates_temp.values))
        x = ro.FloatVector(data)
        if self.params_class is GevParams:
            if self.fit_method == MarginFitMethod.is_mev_gev_fit:
                return self.ismev_gev_fit(x, df_coordinates_temp)
            elif self.fit_method in FEVD_MARGIN_FIT_METHODS:
                return self.extremes_fevd_fit(x, df_coordinates_temp)
            else:
                raise NotImplementedError
        elif self.params_class is ExpParams:
            return self.extreme_fevd_mle_exp_fit(x, df_coordinates_temp)
        else:
            raise NotImplementedError

    # Gev Fit with isMev package

    def ismev_gev_fit(self, x, df_coordinates_temp) -> ResultFromIsmev:
        y = df_coordinates_temp.values
        res = safe_run_r_estimator(function=r('gev.fit'),
                                   xdat=x, y=y, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.margin_function.param_name_to_dims)

    # Gev fit with extRemes package

    def extremes_fevd_fit(self, x, df_coordinates_temp) -> AbstractResultFromExtremes:
        assert self.fit_method in FEVD_MARGIN_FIT_METHODS
        if self.fit_method == MarginFitMethod.extremes_fevd_bayesian:
            return self.extremes_fevd_bayesian_fit(x, df_coordinates_temp)
        else:
            return self.run_fevd_fixed(df_coordinates_temp=df_coordinates_temp,
                                       method=FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR[self.fit_method], x=x)

    def extreme_fevd_mle_exp_fit(self, x, df_coordinates_temp):
        return self.run_fevd_fixed(df_coordinates_temp, "Exponential", x)

    def run_fevd_fixed(self, df_coordinates_temp, method, x):
        if self.fit_method == MarginFitMethod.extremes_fevd_l_moments:
            assert self.margin_function.is_a_stationary_model
        r_type_argument_kwargs, y = self.extreme_arguments(df_coordinates_temp)
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   type=self.type_for_mle,
                                   method=method,
                                   **r_type_argument_kwargs
                                   )
        return ResultFromMleExtremes(res, self.margin_function.param_name_to_dims,
                                     type_for_mle=self.type_for_mle)

    def extremes_fevd_bayesian_fit(self, x, df_coordinates_temp) -> AbstractResultFromExtremes:
        r_type_argument_kwargs, y = self.extreme_arguments(df_coordinates_temp)
        params_initial_fit = self.params_initial_fit_bayesian if self.params_initial_fit_bayesian is not None else {}
        r_type_argument_kwargs['initial'] = r.list(**params_initial_fit)
        # Assert for any non-stationary model that the shape parameter is constant
        # (because the prior function considers that the last parameter should be the shape)
        assert GevParams.SHAPE not in self.margin_function.param_name_to_dims \
               or len(self.margin_function.param_name_to_dims[GevParams.SHAPE]) == 1
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   method='Bayesian',
                                   priorFun="fevdPriorCustom",
                                   priorParams=r.list(q=r.c(6), p=r.c(9)),
                                   iter=self.nb_iterations_for_bayesian_fit,
                                   **r_type_argument_kwargs
                                   )
        return ResultFromBayesianExtremes(res, self.margin_function.param_name_to_dims)

    def extreme_arguments(self, df_coordinates_temp):
        # Disable the use of log sigma parametrization
        r_type_argument_kwargs = {'use.phi': False,
                                  'verbose': False}
        # Load parameters
        r_type_argument_kwargs.update(get_margin_formula_extremes(self.margin_function.form_dict))
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
