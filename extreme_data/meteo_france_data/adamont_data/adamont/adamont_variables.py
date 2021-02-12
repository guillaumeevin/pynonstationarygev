import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable
from root_utils import classproperty


class AbstractAdamontVariable(AbstractVariable):

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return cls.keyword()

    @classmethod
    def indicator_name_for_maxima(cls):
        raise NotImplementedError

class SafranSnowfallSimulationVariable(AbstractAdamontVariable):
    UNIT = SafranSnowfallVariable.UNIT
    NAME = SafranSnowfallVariable.NAME

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return 'Snow'

    @classproperty
    def indicator_name_for_maxima(cls):
        return 'max-1day-snowf'
