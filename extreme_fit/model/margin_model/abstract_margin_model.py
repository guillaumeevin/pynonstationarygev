from abc import ABC

import numpy as np
import pandas as pd
from cached_property import cached_property

from extreme_fit.model.abstract_model import AbstractModel
from extreme_fit.function.margin_function.abstract_margin_function \
    import AbstractMarginFunction
from extreme_fit.model.result_from_model_fit.abstract_result_from_model_fit import AbstractResultFromModelFit
from extreme_fit.model.utils import r
from extreme_fit.distribution.gev.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractMarginModel(AbstractModel, ABC):
    """
    An AbstractMarginModel has one AbstractMarginFunction attribute:
        -margin_function
    """

    def __init__(self, coordinates: AbstractCoordinates, params_user=None, params_class=GevParams):
        super().__init__(params_user)
        assert isinstance(coordinates, AbstractCoordinates), type(coordinates)
        self.coordinates = coordinates
        self.params_class = params_class

    @cached_property
    def margin_function(self) -> AbstractMarginFunction:
        margin_function = self.load_margin_function()
        assert margin_function is not None
        return margin_function

    def load_margin_function(self):
        raise NotImplementedError

    # Conversion class methods

    @classmethod
    def convert_maxima(cls, convertion_r_function, maxima: np.ndarray, coordinates_values: np.ndarray,
                       margin_function: AbstractMarginFunction) -> np.ndarray:
        assert isinstance(coordinates_values, np.ndarray)
        assert len(maxima) == len(coordinates_values)
        converted_maxima = []
        for x, coordinate in zip(maxima, coordinates_values):
            gev_params = margin_function.get_params(coordinate)
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
        maxima_gev = self.frech2gev(maxima_frech, coordinates_values, self.margin_function)
        return maxima_gev

    def rmargin_from_nb_obs(self, nb_obs: int, coordinates_values: np.ndarray,
                            sample_r_function='rgev') -> np.ndarray:
        maxima_gev = []
        for coordinate in coordinates_values:
            gev_params = self.margin_function.get_params(coordinate)
            x_gev = r(sample_r_function)(nb_obs, **gev_params.to_dict())
            assert not np.isnan(x_gev).any(), 'params={} generated Nan values'.format(gev_params.__str__())
            maxima_gev.append(x_gev)
        return np.array(maxima_gev)

    # Fitting methods needs to be defined in child classes

    def fitmargin_from_maxima_gev(self, maxima_gev: np.ndarray, df_coordinates_spatial: pd.DataFrame,
                                  df_coordinates_temporal: pd.DataFrame) -> AbstractResultFromModelFit:
        raise NotImplementedError


