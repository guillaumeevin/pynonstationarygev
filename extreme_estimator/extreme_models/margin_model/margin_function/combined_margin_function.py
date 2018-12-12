from typing import List
from itertools import combinations

import numpy as np

from extreme_estimator.extreme_models.margin_model.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from extreme_estimator.gev_params import GevParams
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class CombinedMarginFunction(AbstractMarginFunction):

    def __init__(self, coordinates: AbstractCoordinates, margin_functions: List[AbstractMarginFunction]):
        super().__init__(coordinates)
        self.margin_functions = margin_functions  # type: List[AbstractMarginFunction]

    def get_gev_params(self, coordinate: np.ndarray) -> GevParams:
        gev_params_list = [margin_function.get_gev_params(coordinate) for margin_function in self.margin_functions]
        mean_gev_params = np.mean(np.array([gev_param.to_array() for gev_param in gev_params_list]), axis=0)
        gev_param = GevParams(*mean_gev_params)
        return gev_param

    @classmethod
    def from_margin_functions(cls, margin_functions: List[AbstractMarginFunction]):
        assert len(margin_functions) > 0
        assert all([isinstance(margin_function, AbstractMarginFunction) for margin_function in margin_functions])
        all_coordinates = [margin_function.coordinates for margin_function in margin_functions]
        for coordinates1, coordinates2 in combinations(all_coordinates, 2):
            assert coordinates1 == coordinates2
        coordinates = all_coordinates[0]
        return cls(coordinates, margin_functions)
