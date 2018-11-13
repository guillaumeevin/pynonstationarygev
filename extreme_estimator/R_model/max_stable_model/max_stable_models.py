from enum import Enum

from extreme_estimator.R_model.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel, \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction


class Smith(AbstractMaxStableModel):

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit=params_start_fit, params_sample=params_sample)
        self.cov_mod = 'gauss'
        self.default_params_start_fit = {
            'cov11': 1,
            'cov12': 0,
            'cov22': 1
        }
        self.default_params_sample = self.default_params_start_fit.copy()


class BrownResnick(AbstractMaxStableModel):

    def __init__(self, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit=params_start_fit, params_sample=params_sample)
        self.cov_mod = 'brown'
        self.default_params_start_fit = {
            'range': 3,
            'smooth': 0.5,
        }
        self.default_params_sample = {
            'range': 3,
            'smooth': 0.5,
        }


class Schlather(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample, covariance_function)
        self.cov_mod = self.covariance_function.name
        self.default_params_sample.update({})
        self.default_params_start_fit = self.default_params_sample.copy()


class Geometric(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample, covariance_function)
        self.cov_mod = 'g' + self.covariance_function.name
        self.default_params_sample .update({'sigma2': 0.5})
        self.default_params_start_fit = self.default_params_sample.copy()


class ExtremalT(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample, covariance_function)
        self.cov_mod = 't' + self.covariance_function.name
        self.default_params_sample .update({'DoF': 2})
        self.default_params_start_fit = self.default_params_sample.copy()


class ISchlather(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, params_start_fit=None, params_sample=None, covariance_function: CovarianceFunction = None):
        super().__init__(params_start_fit, params_sample, covariance_function)
        self.cov_mod = 'i' + self.covariance_function.name
        self.default_params_sample .update({'alpha': 0.5})
        self.default_params_start_fit = self.default_params_sample.copy()
