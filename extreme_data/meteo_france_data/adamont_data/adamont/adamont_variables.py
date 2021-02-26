import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable, \
    TotalSnowLoadVariable, CrocusTotalSweVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable
from root_utils import classproperty


class AbstractAdamontVariable(AbstractVariable):

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return cls.keyword()

    @classmethod
    def indicator_name_for_maxima(cls):
        raise NotImplementedError

    @classmethod
    def transform_annual_maxima(cls, annual_maxima):
        return annual_maxima

class SafranSnowfallSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranSnowfallVariable.UNIT
    NAME = SafranSnowfallVariable.NAME

    # For adamont v1

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'

    # For adamont v2

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return 'Snow'

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-snowf'


class CrocusSweSimulationVariable(AbstractAdamontVariable):
    UNIT = CrocusTotalSweVariable.UNIT
    NAME = CrocusTotalSweVariable.NAME

    # For adamont v2

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return cls.indicator_name_for_maxima.replace('-', '_').capitalize()

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'swe-max-winter-11-04-NN'


class CrocusTotalSnowLoadVariable(CrocusSweSimulationVariable):
    NAME = TotalSnowLoadVariable.NAME
    UNIT = TotalSnowLoadVariable.UNIT

    @classmethod
    def transform_annual_maxima(cls, annual_maxima):
        return AbstractSnowLoadVariable.transform_swe_into_snow_load(annual_maxima)


