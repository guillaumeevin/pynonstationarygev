import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable, \
    TotalSnowLoadVariable, CrocusTotalSweVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable
from root_utils import classproperty


class AbstractAdamontVariable(AbstractVariable):

    # Adamont v1

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        raise NotImplementedError

    @classmethod
    def keyword(cls):
        raise NotImplementedError

    @classmethod
    def indicator_name_for_maxima(cls):
        raise NotImplementedError

    @classmethod
    def indicator_name_for_maxima_date(cls):
        raise NotImplementedError

    @classmethod
    def get_folder_name_from_indicator_name(cls, indicator_name):
        return indicator_name.replace('-', '_').capitalize()

    @classmethod
    def transform_annual_maxima(cls, annual_maxima):
        return annual_maxima


class SafranSnowfallSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranSnowfallVariable.UNIT
    NAME = SafranSnowfallVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return 'Snow'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-snowf'


class CrocusSweSimulationVariable(AbstractAdamontVariable):
    UNIT = CrocusTotalSweVariable.UNIT
    NAME = CrocusTotalSweVariable.NAME

    @classmethod
    def variable_folder_name(cls, annual_maxima_date):
        if annual_maxima_date:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima_date)
        else:
            return cls.get_folder_name_from_indicator_name(cls.indicator_name_for_maxima)

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'swe-max-winter-11-04-NN'

    @classproperty
    def indicator_name_for_maxima_date(cls):
        return 'date-swe-max-winter-11-04-NN'


class CrocusTotalSnowLoadVariable(CrocusSweSimulationVariable):
    NAME = TotalSnowLoadVariable.NAME
    UNIT = TotalSnowLoadVariable.UNIT

    @classmethod
    def transform_annual_maxima(cls, annual_maxima):
        return AbstractSnowLoadVariable.transform_swe_into_snow_load(annual_maxima)
