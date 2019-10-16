import numpy as np
from rpy2 import robjects



class AbstractResultFromModelFit(object):

    def __init__(self, result_from_fit: robjects.ListVector) -> None:
        if hasattr(result_from_fit, 'names'):
            self.name_to_value = {name: result_from_fit.rx2(name) for name in result_from_fit.names}
        else:
            self.name_to_value = {}

    @property
    def names(self):
        return self.name_to_value.keys()

    @property
    def all_parameters(self):
        raise NotImplementedError

    @property
    def margin_coef_dict(self):
        raise NotImplementedError

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





