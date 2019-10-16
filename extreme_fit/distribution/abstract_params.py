from abc import ABC
from typing import List

import numpy as np
import pandas as pd


class AbstractParams(object):
    # Parameters
    PARAM_NAMES = []
    # Quantile
    QUANTILE_10 = 'quantile 0.9'
    QUANTILE_100 = 'quantile 0.99'
    QUANTILE_1000 = 'quantile 0.999'
    QUANTILE_NAMES = [QUANTILE_10, QUANTILE_100, QUANTILE_1000][:-1]
    QUANTILE_P_VALUES = [0.9, 0.99, 0.999][:-1]
    QUANTILE_COLORS = ['orange', 'red', 'darkviolet']
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + QUANTILE_NAMES
    # Extreme parameters
    SCALE = 'scale'
    LOC = 'loc'
    SHAPE = 'shape'

    def __init__(self, loc: float, scale: float, shape: float):
        self.location = loc
        self.scale = scale
        self.shape = shape
        # By default, scale cannot be negative
        # (sometimes it happens, when we want to find a quantile for every point of a 2D map
        # then it can happen that a corner point that was not used for fitting correspond to a negative scale,
        # in the case we set all the parameters as equal to np.nan, and we will not display those points)
        self.has_undefined_parameters = self.scale <= 0

    @classmethod
    def from_dict(cls, params: dict):
        return cls(**params)

    # Parameters

    @property
    def param_values(self) -> List[float]:
        raise NotImplementedError

    def to_dict(self) -> dict:
        assert len(self.PARAM_NAMES) == len(self.param_values)
        return dict(zip(self.PARAM_NAMES, self.param_values))

    def to_serie(self) -> pd.Series:
        return pd.Series(self.to_dict(), index=self.PARAM_NAMES)

    def to_array(self) -> np.ndarray:
        return self.to_serie().values

    # Quantile

    def quantile(self, p) -> float:
        raise NotImplementedError

    @property
    def quantile_name_to_p(self) -> dict:
        return dict(zip(self.QUANTILE_NAMES, self.QUANTILE_P_VALUES))

    @property
    def quantile_name_to_value(self) -> dict:
        return {quantile_name: self.quantile(p) for quantile_name, p in self.quantile_name_to_p.items()}

    # Summary (i.e. parameters & quantiles)

    @property
    def summary_dict(self) -> dict:
        return {**self.to_dict(), **self.quantile_name_to_value}

    @property
    def summary_serie(self) -> pd.Series:
        return pd.Series(self.summary_dict, index=self.SUMMARY_NAMES)

    # Estimators
