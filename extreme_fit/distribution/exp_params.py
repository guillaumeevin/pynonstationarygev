import numpy as np

from extreme_fit.distribution.abstract_params import AbstractParams
from extreme_fit.model.utils import r


class ExpParams(AbstractParams):
    PARAM_NAMES = [AbstractParams.RATE]

    def __init__(self, rate) -> None:
        self.rate = rate
        # todo: is this really the best solution, it might be best to raise an assert
        self.has_undefined_parameters = self.rate < 0

    def quantile(self, p) -> float:
        return r.qexp(p, self.rate)

    @property
    def param_values(self):
        if self.has_undefined_parameters:
            return [np.nan for _ in range(1)]
        else:
            return [self.rate]
