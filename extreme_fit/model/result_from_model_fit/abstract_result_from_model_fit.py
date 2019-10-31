import numpy as np
from rpy2 import robjects



class AbstractResultFromModelFit(object):

    def __init__(self, result_from_fit: robjects.ListVector) -> None:
        if hasattr(result_from_fit, 'names'):
            self.name_to_value = self.get_python_dictionary(result_from_fit)
        else:
            self.name_to_value = {}

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
        raise NotImplementedError

    @property
    def convergence(self) -> str:
        raise NotImplementedError

    @property
    def covariance(self):
        raise NotImplementedError





