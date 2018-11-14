import numpy as np


class GevParams(object):
    GEV_SCALE = 'scale'
    GEV_LOC = 'loc'
    GEV_SHAPE = 'shape'
    GEV_PARAM_NAMES = [GEV_LOC, GEV_SCALE, GEV_SHAPE]

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**params)

    def to_dict(self) -> dict:
        return {
            self.GEV_LOC: self.location,
            self.GEV_SCALE: self.scale,
            self.GEV_SHAPE: self.shape,
        }

    def to_array(self) -> np.ndarray:
        gev_param_name_to_value = self.to_dict()
        return np.array([gev_param_name_to_value[gev_param_name] for gev_param_name in self.GEV_PARAM_NAMES])
