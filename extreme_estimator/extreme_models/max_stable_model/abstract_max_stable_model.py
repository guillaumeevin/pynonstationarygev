import pandas as pd
from enum import Enum

import numpy as np
import rpy2
import rpy2.robjects as ro
from rpy2.robjects import ListVector

from extreme_estimator.extreme_models.abstract_model import AbstractModel


class AbstractMaxStableModel(AbstractModel):

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)
        self.cov_mod = None

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    def fitmaxstab(self, maxima_frech: np.ndarray, df_coordinates: pd.DataFrame, fit_marge=False,
                   fit_marge_form_dict=None, margin_start_dict=None):
        assert isinstance(maxima_frech, np.ndarray)
        assert isinstance(df_coordinates, pd.DataFrame)
        if fit_marge:
            assert fit_marge_form_dict is not None
            assert margin_start_dict is not None

        # Add the colnames to df_coordinates DataFrame to enable specification of the margin functions
        df_coordinates = df_coordinates.copy()
        df_coordinates.colnames = ro.StrVector(list(df_coordinates.columns))
        # Transform the formula string representation into robjects.Formula("y ~ x")
        #  Specify the fit params
        fit_params = self.cov_mod_param.copy()
        start_dict = self.params_start_fit
        if fit_marge:
            start_dict.update(margin_start_dict)
            fit_params.update({k: ro.Formula(v) for k, v in fit_marge_form_dict.items()})
        fit_params['start'] = self.r.list(**start_dict)
        fit_params['fit.marge'] = fit_marge
        # Run the fitmaxstab in R
        # todo: find how to specify the optim function to use
        coord = df_coordinates.values

        try:
            res = self.r.fitmaxstab(data=np.transpose(maxima_frech), coord=coord, **fit_params)  # type: ListVector
        except rpy2.rinterface.RRuntimeError as error:
            raise Exception('Some R exception have been launched at RunTime: \n {}'.format(error.__repr__()))
        # todo: maybe if the convergence was not successful I could try other starting point several times
        # Retrieve the resulting fitted values
        fitted_values = res.rx2('fitted.values')
        fitted_values = {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
        return fitted_values

    def rmaxstab(self, nb_obs: int, coordinates: np.ndarray) -> np.ndarray:
        """
        Return an numpy of maxima. With rows being the stations and columns being the years of maxima
        """
        maxima_frech = np.array(
            self.r.rmaxstab(nb_obs, coordinates, *list(self.cov_mod_param.values()), **self.params_sample))
        return np.transpose(maxima_frech)


class CovarianceFunction(Enum):
    whitmat = 0
    cauchy = 1
    powexp = 2
    bessel = 3


class AbstractMaxStableModelWithCovarianceFunction(AbstractMaxStableModel):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample)
        assert covariance_function is not None
        self.covariance_function = covariance_function
        self.default_params_sample = {
            'range': 3,
            'smooth': 0.5,
            'nugget': 0.5
        }
