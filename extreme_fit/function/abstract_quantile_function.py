import numpy as np

from extreme_fit.function.abstract_function import AbstractFunction
from extreme_fit.function.margin_function.abstract_margin_function import AbstractMarginFunction
import matplotlib.pyplot as plt

from extreme_fit.function.param_function.param_function import AbstractParamFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractQuantileFunction(AbstractFunction):

    def get_quantile(self, coordinate: np.ndarray, is_transformed: bool = True) -> float:
        transformed_coordinate = coordinate if is_transformed else self.transform(coordinate)
        return self._get_quantile(transformed_coordinate)

    def _get_quantile(self, coordinate: np.ndarray):
        raise NotImplementedError

    def visualize(self, show=True):
        if self.coordinates.nb_coordinates == 1:
            self.visualize_1D(show=show)
        elif self.coordinates.nb_coordinates == 2:
            self.visualize_2D()
        else:
            return
            # raise NotImplementedError

    def visualize_1D(self, ax=None, show=True):
        if ax is None:
            ax = plt.gca()
        x = self.coordinates.coordinates_values()
        resolution = 100
        x = np.linspace(x.min(), x.max(), resolution)
        y = [self.get_quantile(np.array([e])) for e in x]
        ax.plot(x, y)
        if show:
            plt.show()

    def visualize_2D(self):
        return


class QuantileFunctionFromParamFunction(AbstractQuantileFunction):

    def __init__(self, coordinates: AbstractCoordinates, param_function: AbstractParamFunction):
        super().__init__(coordinates)
        self.param_function = param_function

    def _get_quantile(self, coordinate: np.ndarray) -> float:
        return self.param_function.get_param_value(coordinate)


class QuantileFunctionFromMarginFunction(AbstractQuantileFunction):

    def __init__(self, coordinates: AbstractCoordinates, margin_function: AbstractMarginFunction, quantile: float):
        super().__init__(coordinates)
        self.margin_function = margin_function
        self.quantile = quantile

    def _get_quantile(self, coordinate: np.ndarray) -> float:
        gev_params = self.margin_function.get_gev_params(coordinate)
        return gev_params.quantile(self.quantile)
