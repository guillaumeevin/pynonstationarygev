import numpy as np
import pandas as pd

from extreme_estimator.extreme_models.utils import r


class GevParams(object):
    # GEV parameters
    GEV_SCALE = 'scale'
    GEV_LOC = 'loc'
    GEV_SHAPE = 'shape'
    GEV_PARAM_NAMES = [GEV_LOC, GEV_SCALE, GEV_SHAPE]
    # GEV quantiles
    GEV_QUANTILE_10 = 'quantile 10'
    GEV_QUANTILE_100 = 'quantile 100'
    GEV_QUANTILE_1000 = 'quantile 1000'
    GEV_QUANTILE_NAMES = [GEV_QUANTILE_10, GEV_QUANTILE_100, GEV_QUANTILE_1000]
    # GEV values
    GEV_VALUE_NAMES = GEV_PARAM_NAMES + GEV_QUANTILE_NAMES[:-1]

    # GEV parameters

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape
        # self.scale = max(self.scale, 1e-4)
        assert self.scale > 0

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
        return self.to_serie().values

    def to_serie(self) -> pd.Series:
        return pd.Series(self.to_dict(), index=self.GEV_PARAM_NAMES)

    # GEV quantiles

    def qgev(self, p) -> float:
        return r.qgev(p, self.location, self.scale, self.shape)[0]

    @property
    def quantile_name_to_p(self) -> dict:
        return {
            self.GEV_QUANTILE_10: 0.1,
            self.GEV_QUANTILE_100: 0.01,
            self.GEV_QUANTILE_1000: 0.001,
        }

    @property
    def quantile_dict(self) -> dict:
        return {quantile_name: self.qgev(p) for quantile_name, p in self.quantile_name_to_p.items()}

    # GEV values

    @property
    def value_dict(self) -> dict:
        return {**self.to_dict(), **self.quantile_dict}

    @property
    def value_serie(self) -> pd.Series:
        return pd.Series(self.value_dict, index=self.GEV_VALUE_NAMES)


