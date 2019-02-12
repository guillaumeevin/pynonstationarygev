from typing import List

import numpy as np
import pandas as pd


class AbstractParams(object):
    # Parameters
    PARAM_NAMES = []
    # Quantile
    QUANTILE_10 = 'quantile 10'
    QUANTILE_100 = 'quantile 100'
    QUANTILE_1000 = 'quantile 1000'
    QUANTILE_NAMES = [QUANTILE_10, QUANTILE_100, QUANTILE_1000][:-1]
    QUANTILE_P_VALUES = [0.9, 0.99, 0.999]
    # Summary
    SUMMARY_NAMES = PARAM_NAMES + QUANTILE_NAMES

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

