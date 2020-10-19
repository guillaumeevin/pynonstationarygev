import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class AbstractAdamontVariable(AbstractVariable):

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return cls.keyword()


class SafranSnowfallSimulationVariable(AbstractAdamontVariable):
    UNIT = 'kg $m^{-2}$'

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'

    @classmethod
    def variable_name_for_folder_and_nc_file(cls):
        return 'Snow'
