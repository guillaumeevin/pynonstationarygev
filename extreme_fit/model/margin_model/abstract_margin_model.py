from abc import ABC

import numpy as np
import pandas as pd

from extreme_fit.model.abstract_model import AbstractModel
from extreme_fit.model.margin_model.margin_function.abstract_margin_function \
    import AbstractMarginFunction
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.utils import r
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMarginModel(AbstractModel, ABC):
    """
    An AbstractMarginModel has two main AbstractMarginFunction attributes:
        -margin_function_sample for sampling
        -margin_function_start_fit for starting to fit
    """

    def __init__(self, coordinates: AbstractCoordinates, use_start_value=False,
                 params_start_fit=None, params_sample=None):
        super().__init__(use_start_value, params_start_fit, params_sample)
        assert isinstance(coordinates, AbstractCoordinates), type(coordinates)
        self.coordinates = coordinates
        self.margin_function_sample = None  # type: AbstractMarginFunction
        self.margin_function_start_fit = None  # type: AbstractMarginFunction
        self.load_margin_functions()

    def load_margin_functions(self):
        raise NotImplementedError

    def default_load_margin_functions(self, margin_function_class):
        self.margin_function_sample = margin_function_class(coordinates=self.coordinates,
                                                            default_params=GevParams.from_dict(self.params_sample))
        self.margin_function_start_fit = margin_function_class(coordinates=self.coordinates,
                                                               default_params=GevParams.from_dict(
                                                                   self.params_start_fit))

    # Conversion class methods

    @classmethod
    def convert_maxima(cls, convertion_r_function, maxima: np.ndarray, coordinates_values: np.ndarray,
                       margin_function: AbstractMarginFunction) -> np.ndarray:
        assert isinstance(coordinates_values, np.ndarray)
        assert len(maxima) == len(coordinates_values)
        converted_maxima = []
        for x, coordinate in zip(maxima, coordinates_values):
            gev_params = margin_function.get_gev_params(coordinate)
            x_gev = convertion_r_function(x, **gev_params.to_dict())
            converted_maxima.append(x_gev)
        return np.array(converted_maxima)

    @classmethod
    def gev2frech(cls, maxima_gev: np.ndarray, coordinates_values: np.ndarray,
                  margin_function: AbstractMarginFunction) -> np.ndarray:
        return cls.convert_maxima(r.gev2frech, maxima_gev, coordinates_values, margin_function)

    @classmethod
    def frech2gev(cls, maxima_frech: np.ndarray, coordinates_values: np.ndarray,
                  margin_function: AbstractMarginFunction) -> np.ndarray:
        return cls.convert_maxima(r.frech2gev, maxima_frech, coordinates_values, margin_function)

    # Sampling methods

    def rmargin_from_maxima_frech(self, maxima_frech: np.ndarray, coordinates_values: np.ndarray) -> np.ndarray:
        maxima_gev = self.frech2gev(maxima_frech, coordinates_values, self.margin_function_sample)
        return maxima_gev

    def rmargin_from_nb_obs(self, nb_obs: int, coordinates_values: np.ndarray) -> np.ndarray:
        maxima_gev = []
        for coordinate in coordinates_values:
            gev_params = self.margin_function_sample.get_gev_params(coordinate)
            x_gev = r.rgev(nb_obs, **gev_params.to_dict())
            maxima_gev.append(x_gev)
        return np.array(maxima_gev)

    # Fitting methods needs to be defined in child classes

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, df_coordinates_spatial: pd.DataFrame,
                                  df_coordinates_temporal: pd.DataFrame) -> AbstractResultFromModelFit:
        raise NotImplementedError

