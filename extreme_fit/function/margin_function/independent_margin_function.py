from typing import Dict, Union

import numpy as np

from extreme_fit.function.param_function.param_function import AbstractParamFunction
from extreme_fit.distribution.gev.gev_params import GevParams
from extreme_fit.function.margin_function.abstract_margin_function import \
    AbstractMarginFunction
from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class IndependentMarginFunction(AbstractMarginFunction):
    """
        IndependentMarginFunction: each parameter of the GEV are modeled independently
    """

    def __init__(self, coordinates: AbstractCoordinates, params_class: type = GevParams):
        """Attribute 'param_name_to_param_function' maps each GEV parameter to its corresponding function"""
        super().__init__(coordinates, params_class)
        self.param_name_to_param_function = None  # type: Union[None, Dict[str, AbstractParamFunction]]

    def get_params(self, coordinate: np.ndarray, is_transformed: bool = True) -> GevParams:
        """Each GEV parameter is computed independently through its corresponding param_function"""
        # Since all the coordinates are usually transformed by default
        # then we assume that the input coordinate are transformed by default
        assert self.param_name_to_param_function is not None
        assert len(self.param_name_to_param_function) == len(self.params_class.PARAM_NAMES)
        transformed_coordinate = coordinate if is_transformed else self.transform(coordinate)
        params = {param_name: param_function.get_param_value(transformed_coordinate)
                  for param_name, param_function in self.param_name_to_param_function.items()}
        return self.params_class.from_dict(params)





