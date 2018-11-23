import pandas as pd
from enum import Enum

import numpy as np
import rpy2
import rpy2.robjects as robjects

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

        # Prepare the data and the coord objects
        data = np.transpose(maxima_frech)
        coord = robjects.vectors.Matrix(df_coordinates.values)
        coord.colnames = robjects.StrVector(list(df_coordinates.columns))

        #  Prepare the fit params
        fit_params = self.cov_mod_param.copy()
        start_dict = self.params_start_fit
        # Remove the 'var' parameter from the start_dict in the 2D case, otherwise fitmaxstab crashes
        if len(df_coordinates.columns) == 2 and 'var' in start_dict.keys():
                start_dict.pop('var')
        if fit_marge:
            start_dict.update(margin_start_dict)
            fit_params.update({k: robjects.Formula(v) for k, v in fit_marge_form_dict.items()})
        fit_params['start'] = self.r.list(**start_dict)
        fit_params['fit.marge'] = fit_marge

        # Run the fitmaxstab in R
        try:
            res = self.r.fitmaxstab(data=data, coord=coord, **fit_params)  # type: robjects.ListVector
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
