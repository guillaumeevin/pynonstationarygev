import numpy as np
from cached_property import cached_property

from extreme_fit.function.param_function.param_function import LinearParamFunction
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit


class ResultFromQuantreg(AbstractResultFromModelFit):

    @property
    def coefficients(self):
        return np.array(self.name_to_value['coefficients'])
