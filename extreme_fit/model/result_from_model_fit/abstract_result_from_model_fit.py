import numpy as np
from rpy2 import robjects


class AbstractResultFromModelFit(object):

    def __init__(self, result_from_fit: robjects.ListVector) -> None:
        self.result_from_fit = result_from_fit
        if hasattr(result_from_fit, 'names'):
            self.name_to_value = self.get_python_dictionary(result_from_fit)
        else:
            self.name_to_value = {}

    @property
    def variance_covariance_matrix(self):
        raise NotImplementedError

    @property
    def standard_errors_for_mle(self):
        """ See Coles 2001 page 41, for an example"""
        return np.sqrt(np.diagonal(self.variance_covariance_matrix))

    @staticmethod
    def get_python_dictionary(r_dictionary):
        return {name: r_dictionary.rx2(name) for name in r_dictionary.names}

    @property
    def names(self):
        return self.name_to_value.keys()

    @property
    def all_parameters(self):
        raise NotImplementedError

    @property
    def log_scale(self):
        # todo: refactor, put raise NotImplementError, and fix the subfunctions for the other Result objects
        return None

    @property
    def param_name_to_name_of_the_climatic_effects_to_load_margin_function(self):
        # todo: refactor, put raise NotImplementError, and fix the subfunctions for the other Result objects
        return None

    @property
    def param_name_to_climate_coordinates_with_effects_to_load_margin_function(self):
        return None

    @property
    def margin_coef_ordered_dict(self):
        raise NotImplementedError

    @property
    def margin_coef_ordered_names(self):
        return list(self.margin_coef_ordered_dict.keys())

    @property
    def other_coef_dict(self):
        raise NotImplementedError

    @property
    def nllh(self):
        raise NotImplementedError

    @property
    def deviance(self):
        return 2 * self.nllh

    @property
    def bic(self):
        raise NotImplementedError

    @property
    def aic(self):
        raise NotImplementedError

    @property
    def convergence(self) -> str:
        raise NotImplementedError

    @property
    def covariance(self):
        raise NotImplementedError
