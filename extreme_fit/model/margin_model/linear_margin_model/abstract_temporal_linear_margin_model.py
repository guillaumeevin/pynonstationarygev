import numpy as np
import pandas as pd
from rpy2 import robjects

from extreme_fit.distribution.exp_params import ExpParams
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.model.margin_model.linear_margin_model.linear_margin_model import LinearMarginModel
from extreme_fit.model.margin_model.utils import MarginFitMethod, fitmethod_to_str, FEVD_MARGIN_FIT_METHODS, \
    FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_bayesian_extremes import \
    AbstractResultFromExtremes, ResultFromBayesianExtremes
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_evgam import ResultFromEvgam
from extreme_fit.model.result_from_model_fit.result_from_extremes.result_from_mle_extremes import ResultFromMleExtremes
from extreme_fit.model.result_from_model_fit.result_from_ismev import ResultFromIsmev
from extreme_fit.model.utils import r, ro, get_null, get_margin_formula_extremes, get_r_dataframe_from_python_dataframe
from extreme_fit.model.utils import safe_run_r_estimator
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractTemporalLinearMarginModel(LinearMarginModel):
    """Linearity only with respect to the temporal coordinates"""

    def __init__(self, coordinates: AbstractCoordinates,
                 params_user=None, starting_point=None,
                 fit_method=MarginFitMethod.is_mev_gev_fit,
                 nb_iterations_for_bayesian_fit=5000,
                 params_initial_fit_bayesian=None,
                 type_for_MLE="GEV",
                 params_class=GevParams,
                 temporal_covariate_for_fit=None):
        super().__init__(coordinates, params_user, starting_point, params_class, fit_method, temporal_covariate_for_fit)
        self.type_for_mle = type_for_MLE
        self.params_initial_fit_bayesian = params_initial_fit_bayesian
        self.nb_iterations_for_bayesian_fit = nb_iterations_for_bayesian_fit
        assert isinstance(self.fit_method, MarginFitMethod), self.fit_method

    def fitmargin_from_maxima_gev(self, data: np.ndarray, df_coordinates_spat: pd.DataFrame,
                                  df_coordinates_temp: pd.DataFrame) -> AbstractResultFromModelFit:
        data = data.flatten()
        assert len(df_coordinates_temp) == len(data)
        if not (df_coordinates_spat is None or df_coordinates_spat.empty):
            assert len(df_coordinates_spat) == len(data)

        x = ro.FloatVector(data)
        if self.params_class is GevParams:
            if self.fit_method == MarginFitMethod.is_mev_gev_fit:
                return self.ismev_gev_fit(x, df_coordinates_temp)
            elif self.fit_method in FEVD_MARGIN_FIT_METHODS:
                return self.extremes_fevd_fit(x, df_coordinates_temp, df_coordinates_spat)
            elif self.fit_method is MarginFitMethod.evgam:
                return self.extremes_evgam_fit(x, df_coordinates_temp, df_coordinates_spat)
            else:
                raise NotImplementedError
        elif self.params_class is ExpParams:
            return self.extreme_fevd_mle_exp_fit(x, df_coordinates_temp, df_coordinates_spat)
        else:
            raise NotImplementedError

    # Gev Fit with isMev package

    def ismev_gev_fit(self, x, df_coordinates_temp) -> ResultFromIsmev:
        y = df_coordinates_temp.values
        res = safe_run_r_estimator(function=r('gev.fit'),
                                   xdat=x, y=y, mul=self.mul,
                                   sigl=self.sigl, shl=self.shl)
        return ResultFromIsmev(res, self.param_name_to_list_for_result)

    # Gev fit with evgam

    def extremes_evgam_fit(self, x, df_coordinates_temp, df_coordinates_spat) -> AbstractResultFromExtremes:
        margin_formula = get_margin_formula_extremes(self.margin_function.form_dict, transformed_as_formula=False)
        maxima_column_name = 'Maxima'
        formula_list = [maxima_column_name + " " + v if i == 0 else v for i, v in enumerate(margin_formula.values())]
        formula = r.list(*[robjects.Formula(f) for f in formula_list])
        df = pd.DataFrame({maxima_column_name: np.array(x)})
        df = pd.concat([df, df_coordinates_spat, df_coordinates_temp], axis=1)
        assert not df.isnull().any(axis=1).any(), "Some Nan values in df:\n {}".format(df)
        data = get_r_dataframe_from_python_dataframe(df)
        if self.type_for_mle is not "GEV":
            raise NotImplementedError
        res = safe_run_r_estimator(function=r('evgam'),
                                   formula=formula,
                                   data=data,
                                   family=self.type_for_mle.lower(),
                                   maxdata=1e10,
                                   )
        return ResultFromEvgam(res, self.param_name_to_list_for_result,
                                     self.coordinates.dim_to_coordinate,
                                     type_for_mle=self.type_for_mle)

    # Gev fit with extRemes package

    def extremes_fevd_fit(self, x, df_coordinates_temp, df_coordinates_spat) -> AbstractResultFromExtremes:
        assert self.fit_method in FEVD_MARGIN_FIT_METHODS
        if self.fit_method == MarginFitMethod.extremes_fevd_bayesian:
            return self.extremes_fevd_bayesian_fit(x, df_coordinates_temp)
        else:
            return self.run_fevd_fixed(df_coordinates_temp=df_coordinates_temp,
                                       df_coordinates_spat=df_coordinates_spat,
                                       method=FEVD_MARGIN_FIT_METHOD_TO_ARGUMENT_STR[self.fit_method], x=x)

    def extreme_fevd_mle_exp_fit(self, x, df_coordinates_temp, df_coordinates_spat):
        return self.run_fevd_fixed(df_coordinates_temp, df_coordinates_spat, "Exponential", x)

    def run_fevd_fixed(self, df_coordinates_temp, df_coordinates_spat, method, x):
        if self.fit_method == MarginFitMethod.extremes_fevd_l_moments:
            assert self.margin_function.is_a_stationary_model
        r_type_argument_kwargs, y = self.extreme_arguments(df_coordinates_temp, df_coordinates_spat)
        res = safe_run_r_estimator(function=r('fevd_fixed'),
                                   x=x,
                                   data=y,
                                   type=self.type_for_mle,
                                   method=method,
                                   **r_type_argument_kwargs
                                   )
        return ResultFromMleExtremes(res, self.param_name_to_list_for_result,
                                     self.coordinates.dim_to_coordinate,
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
        return ResultFromBayesianExtremes(res, self.param_name_to_list_for_result)

    def extreme_arguments(self, df_coordinates_temp, df_coordinates_spat=None):

        # Load parameters

        if df_coordinates_spat is None or df_coordinates_spat.empty:
            df = df_coordinates_temp
        else:
            df = pd.concat([df_coordinates_spat, df_coordinates_temp], axis=1)
        y = get_r_dataframe_from_python_dataframe(df)

        # Disable the use of log sigma parametrization
        r_type_argument_kwargs = {'use.phi': False,
                                  'verbose': False}
        r_type_argument_kwargs.update(get_margin_formula_extremes(self.margin_function.form_dict))
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

    @property
    def is_gumbel_model(self):
        return self.type_for_mle == "Gumbel"