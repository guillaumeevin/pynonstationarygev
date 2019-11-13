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


class AbstractSnowLoadVariable(CrocusVariable):
    UNIT = 'kN $m^{-2}$'
    snow_load_multiplication_factor = 9.81 / 1000

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.snow_load_multiplication_factor * super().daily_time_serie_array


class RecentSnowLoadVariable(AbstractSnowLoadVariable, CrocusRecentSweVariable):
    NAME = 'Snow load last 3 days'

class TotalSnowLoadVariable(AbstractSnowLoadVariable, CrocusTotalSweVariable):
    NAME = 'Snow load total'


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "DSN_T_ISBA"
