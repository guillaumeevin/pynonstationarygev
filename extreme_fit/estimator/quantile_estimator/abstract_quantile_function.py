import numpy as np

from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction


class AbstractQuantileFunction(object):

    def __init__(self, margin_function: AbstractMarginFunction, quantile: float):
        self.margin_function = margin_function
        self.quantile = quantile

    def get_quantile(self, coordinate: np.ndarray) -> float:
        gev_params = self.margin_function.get_gev_params(coordinate)
        return gev_params.quantile(self.quantile)