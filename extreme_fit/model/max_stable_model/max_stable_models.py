from enum import Enum

from extreme_fit.model.max_stable_model.abstract_max_stable_model import AbstractMaxStableModel, \
    AbstractMaxStableModelWithCovarianceFunction, CovarianceFunction


class Smith(AbstractMaxStableModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_mod = 'gauss'
        self.default_params_start_fit = {
            'var': 1,
            'cov11': 1,
            'cov12': 0,
            'cov22': 1
        }
        self.default_params_sample = self.default_params_start_fit.copy()

    def remove_unused_parameters(self, start_dict, fitmaxstab_with_one_dimensional_data):
        if fitmaxstab_with_one_dimensional_data:
            start_dict = {'cov': start_dict['var']}
        else:
            start_dict.pop('var')
        return start_dict


class BrownResnick(AbstractMaxStableModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_mod = self.covariance_function.name
        self.default_params_sample.update({})
        self.default_params_start_fit = self.default_params_sample.copy()


class Geometric(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_mod = 'g' + self.covariance_function.name
        self.default_params_sample.update({'sigma2': 0.5})
        self.default_params_start_fit = self.default_params_sample.copy()


class ExtremalT(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_mod = 't' + self.covariance_function.name
        self.default_params_sample.update({'DoF': 2})
        self.default_params_start_fit = self.default_params_sample.copy()


class ISchlather(AbstractMaxStableModelWithCovarianceFunction):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cov_mod = 'i' + self.covariance_function.name
        self.default_params_sample.update({'alpha': 0.5})
        self.default_params_start_fit = self.default_params_sample.copy()