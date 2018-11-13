import rpy2
from enum import Enum

from rpy2.robjects import ListVector
from extreme_estimator.R_fit.utils import get_loaded_r
import numpy as np


class AbstractMaxStableModel(object):

    def __init__(self, params_start_fit=None, params_sample=None):
        self.cov_mod = None
        self.default_params_start_fit = None
        self.default_params_sample = None
        self.user_params_start_fit = params_start_fit
        self.user_params_sample = params_sample
        self.r = get_loaded_r()

    def fitmaxstab(self, maxima_normalized: np.ndarray, coord: np.ndarray, fit_marge=False):
        assert all([isinstance(arr, np.ndarray) for arr in [maxima_normalized, coord]])
        #  Specify the fit params
        fit_params = {
            'fit.marge': fit_marge,
            'start': self.r.list(**self.params_start_fit),
        }
        # Run the fitmaxstab in R
        # todo: find how to specify the optim function to use
        try:
            res = self.r.fitmaxstab(np.transpose(maxima_normalized), coord, **self.cov_mod_param, **fit_params)  # type: ListVector
        except rpy2.rinterface.RRuntimeError as error:
            raise Exception('Some R exception have been launched at RunTime: {}'.format(error.__repr__()))
        # todo: maybe if the convergence was not successful I could try other starting point several times
        # Retrieve the resulting fitted values
        fitted_values = res.rx2('fitted.values')
        fitted_values = {key: fitted_values.rx2(key)[0] for key in fitted_values.names}
        return fitted_values

    def rmaxstab(self, nb_obs: int, coord: np.ndarray, ) -> np.ndarray:
        """
        Return an numpy of maxima. With rows being the stations and columns being the years of maxima
        """
        maxima = np.array(self.r.rmaxstab(nb_obs, coord, *list(self.cov_mod_param.values()), **self.params_sample))
        return np.transpose(maxima)

    @property
    def cov_mod_param(self):
        return {'cov.mod': self.cov_mod}

    @property
    def params_start_fit(self):
        return self.merge_params(default_params=self.default_params_start_fit, input_params=self.user_params_start_fit)

    @property
    def params_sample(self):
        return self.merge_params(default_params=self.default_params_sample, input_params=self.user_params_sample)

    @staticmethod
    def merge_params(default_params, input_params):
        assert default_params is not None, 'some default_params need to be specified'
        merged_params = default_params.copy()
        if input_params is not None:
            assert isinstance(default_params, dict) and isinstance(input_params, dict)
            assert set(input_params.keys()).issubset(set(default_params.keys()))
            merged_params.update(input_params)
        return merged_params


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