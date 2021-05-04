import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):
    snow_load_multiplication_factor = 9.81 / 1000

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array


class CrocusTotalSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent total'
    UNIT = 'kg $m^{-2}$'

    @classmethod
    def keyword(cls):
        return 'WSN_T_ISBA'


class CrocusRecentSweVariableOneDay(CrocusTotalSweVariable):
    NAME = 'Snow Water Equivalent last 1 day'

    @classmethod
    def keyword(cls):
        return 'SWE_1DY_ISBA'


class CrocusRecentSweVariableThreeDays(CrocusTotalSweVariable):
    NAME = 'SWE in 3 days'

    @classmethod
    def keyword(cls):
        return 'SWE_3DY_ISBA'


class CrocusRecentSweVariableFiveDays(CrocusTotalSweVariable):
    NAME = 'Snow Water Equivalent last 5 days'

    @classmethod
    def keyword(cls):
        return 'SWE_5DY_ISBA'


class CrocusRecentSweVariableSevenDays(CrocusTotalSweVariable):
    NAME = 'Snow Water Equivalent last 7 days'

    @classmethod
    def keyword(cls):
        return 'SWE_7DY_ISBA'


class CrocusRamsondVariable(CrocusVariable):

    @classmethod
    def keyword(cls):
        return "RAMSOND_ISBA"


class AbstractSnowLoadVariable(CrocusVariable):
    UNIT = 'kN $m^{-2}$'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.transform_swe_into_snow_load(super().daily_time_serie_array)

    @classmethod
    def transform_swe_into_snow_load(cls, swe):
        return cls.snow_load_multiplication_factor * swe


class RecentSnowLoadVariableOneDay(AbstractSnowLoadVariable, CrocusRecentSweVariableOneDay):
    NAME = 'Snow load last 1 day'


class RecentSnowLoadVariableThreeDays(AbstractSnowLoadVariable, CrocusRecentSweVariableThreeDays):
    NAME = 'Snow load last 3 days'


class RecentSnowLoadVariableFiveDays(AbstractSnowLoadVariable, CrocusRecentSweVariableFiveDays):
    NAME = 'Snow load last 5 days'


class RecentSnowLoadVariableSevenDays(AbstractSnowLoadVariable, CrocusRecentSweVariableSevenDays):
    NAME = 'Snow load last 7 days'


class TotalSnowLoadVariable(AbstractSnowLoadVariable, CrocusTotalSweVariable):
    NAME = 'Snow load total'


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "DSN_T_ISBA"


class CrocusDepthIn3DaysVariable(CrocusVariable):
    NAME = 'Snow Depth in 3 days'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "SD_3DY_ISBA"


class CrocusDepthWetVariable(CrocusVariable):
    NAME = 'Wet Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "WET_TH_ISBA"


class CrocusDensityVariable(CrocusVariable):
    NAME = 'Snow Density'
    # UNIT = '$\\textnormal{kg m}^{-3}$'
    UNIT = 'kg $m^{-3}$'

    @classmethod
    def keyword(cls):
        # Load the snow depth by default
        return "DSN_T_ISBA"


class CrocusSnowLoadEurocodeVariable(AbstractSnowLoadVariable, CrocusDepthVariable):
    eurocode_snow_density = 150

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        snow_weight = super(CrocusDepthVariable, self).daily_time_serie_array * self.eurocode_snow_density
        snow_pressure = self.snow_load_multiplication_factor * snow_weight
        return snow_pressure
