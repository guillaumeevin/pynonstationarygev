from typing import Dict, List

from spatio_temporal_dataset.coordinates.abstract_coordinates import AbstractCoordinates


class AbstractCoef(object):

    def __init__(self, gev_param_name: str = '', default_value: float = 0.0, idx_to_coef=None):
        self.gev_param_name = gev_param_name
        self.default_value = default_value
        self.idx_to_coef = idx_to_coef

    def get_coef(self, idx) -> float:
        if self.idx_to_coef is None:
            return self.default_value
        else:
            return self.idx_to_coef.get(idx, self.compute_default_value(idx))

    def compute_default_value(self, idx):
        return self.default_value

    @property
    def intercept(self) -> float:
        return self.default_value

    """ Coef dict """

    def coef_dict(self, dims: List[int], coordinates: AbstractCoordinates) -> Dict[str, float]:
        raise NotImplementedError

    @classmethod
    def from_coef(cls, coef_dict: Dict[str, float], gev_param_name: str, dims: List[int], coordinates: AbstractCoordinates):
        raise NotImplementedError

    """ Form dict """

    def form_dict(self, names: List[str]) -> Dict[str, str]:
        formula_str = '1' if not names else '+'.join(names)
        return {self.gev_param_name + '.form': self.gev_param_name + ' ~ ' + formula_str}