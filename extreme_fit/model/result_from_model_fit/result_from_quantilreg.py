from cached_property import cached_property

from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit


class ResultFromQuantreg(AbstractResultFromModelFit):

    @property
    def coefficients(self):
        return self.name_to_value['coefficients']

    @cached_property
    def quantile_function(self):
        print(self.coefficients)