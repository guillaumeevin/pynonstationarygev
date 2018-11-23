import numpy as np

from extreme_estimator.extreme_models.abstract_model import AbstractModel
from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import AbstractMarginFunction
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMarginModel(AbstractModel):

    def __init__(self, coordinates: AbstractCoordinates, params_start_fit=None, params_sample=None):
        super().__init__(params_start_fit, params_sample)
        self.coordinates = coordinates
        self.margin_function_sample = None  # type: AbstractMarginFunction
        self.margin_function_start_fit = None  # type: AbstractMarginFunction
        self.load_margin_functions()

    def load_margin_functions(self, margin_function_class: type = None):
        assert margin_function_class is not None
        self.margin_function_sample = margin_function_class(coordinates=self.coordinates,
                                                            default_params=GevParams.from_dict(self.params_sample))
        self.margin_function_start_fit = margin_function_class(coordinates=self.coordinates,
                                                               default_params=GevParams.from_dict(
                                                                   self.params_start_fit))

    # Conversion class methods

    @classmethod
    def convert_maxima(cls, convertion_r_function, maxima: np.ndarray, coordinates: np.ndarray,
                       margin_function: AbstractMarginFunction):
        assert isinstance(coordinates, np.ndarray)
        assert len(maxima) == len(coordinates)
        converted_maxima = []
        for x, coordinate in zip(maxima, coordinates):
            gev_params = margin_function.get_gev_params(coordinate)
            x_gev = convertion_r_function(x, **gev_params.to_dict())
            converted_maxima.append(x_gev)
        return np.array(converted_maxima)

    @classmethod
    def gev2frech(cls, maxima_gev: np.ndarray, coordinates: np.ndarray, margin_function: AbstractMarginFunction):
        return cls.convert_maxima(cls.r.gev2frech, maxima_gev, coordinates, margin_function)

    @classmethod
    def frech2gev(cls, maxima_frech: np.ndarray, coordinates: np.ndarray, margin_function: AbstractMarginFunction):
        return cls.convert_maxima(cls.r.frech2gev, maxima_frech, coordinates, margin_function)

    # Sampling methods

    def rmargin_from_maxima_frech(self, maxima_frech: np.ndarray, coordinates: np.ndarray):
        maxima_gev = self.frech2gev(maxima_frech, coordinates, self.margin_function_sample)
        return maxima_gev

    def rmargin_from_nb_obs(self, nb_obs, coordinates):
        maxima_gev = []
        for coordinate in coordinates:
            gev_params = self.margin_function_sample.get_gev_params(coordinate)
            x_gev = self.r.rgev(nb_obs, **gev_params.to_dict())
            maxima_gev.append(x_gev)
        return np.array(maxima_gev)

    # Fitting methods needs to be defined in child classes

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, coordinates: np.ndarray) -> AbstractMarginFunction:
        pass
