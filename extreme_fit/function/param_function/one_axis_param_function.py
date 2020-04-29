import numpy as np


class AbstractOneAxisParamFunction(object):

    def __init__(self, dim: int, coordinates: np.ndarray):
        self.dim = dim
        self.t_min = coordinates[:, dim].min()
        self.t_max = coordinates[:, dim].max()

    def get_param_value(self, coordinate: np.ndarray, assert_range=True) -> float:
        t = coordinate[self.dim]
        if assert_range:
            assert self.t_min <= t <= self.t_max, '{} is out of bounds ({}, {})'.format(t, self.t_min, self.t_max)
        return self._get_param_value(t)

    def _get_param_value(self, t) -> float:
        raise NotImplementedError


class LinearOneAxisParamFunction(AbstractOneAxisParamFunction):

    def __init__(self, dim: int, coordinates: np.ndarray, coef: float = 0.01):
        super().__init__(dim, coordinates)
        self.coef = coef

    def _get_param_value(self, t) -> float:
        return t * self.coef


class QuadraticOneAxisParamFunction(AbstractOneAxisParamFunction):

    def __init__(self, dim: int, coordinates: np.ndarray, coef1: float = 0.01, coef2: float = 0.01):
        super().__init__(dim, coordinates)
        self.coef1 = coef1
        self.coef2 = coef2

    def _get_param_value(self, t) -> float:
        return np.power(t, 2) * self.coef2 + t * self.coef1
