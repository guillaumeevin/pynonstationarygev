import numpy as np

from extreme_fit.distribution.abstract_params import AbstractParams


class ExpParams(AbstractParams):
    PARAM_NAMES = [AbstractParams.RATE]

    def __init__(self, rate) -> None:
        self.rate = rate
        self.has_undefined_parameters = self.rate < 0

    @property
    def param_values(self):
        if self.has_undefined_parameters:
            return [np.nan for _ in range(1)]
        else:
            return [self.rate]
