import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array


class CrocusTotalSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent total'
    UNIT = 'kg $m^{-2}$'

    @classmethod
    def keyword(cls):
        return 'WSN_T_ISBA'


class CrocusRecentSweVariable(CrocusTotalSweVariable):
    NAME = 'Snow Water Equivalent last 3 days'

    @classmethod
    def keyword(cls):
        return 'SWE_3DY_ISBA'


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "DSN_T_ISBA"
