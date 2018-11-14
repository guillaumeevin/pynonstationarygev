from enum import Enum

import numpy as np
import rpy2
from rpy2.robjects import ListVector

from extreme_estimator.R_model.abstract_model import AbstractModel


class AbstractMaxStableModel(AbstractModel):

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)
        self.cov_mod = None

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    def fitmaxstab(self, maxima_frech: np.ndarray, coord: np.ndarray, fit_marge=False):
        assert all([isinstance(arr, np.ndarray) for arr in [maxima_frech, coord]])
        #  Specify the fit params
        fit_params = {
            'fit.marge': fit_marge,
            'start': self.r.list(**self.params_start_fit),
        }
        # Run the fitmaxstab in R
        # todo: find how to specify the optim function to use
        try:
            res = self.r.fitmaxstab(np.transpose(maxima_frech), coord, **self.cov_mod_param,
                                    **fit_params)  # type: ListVector
        except rpy2.rinterface.RRuntimeError as error:
            raise Exception('Some R exception have been launched at RunTime: {}'.format(error.__repr__()))
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
