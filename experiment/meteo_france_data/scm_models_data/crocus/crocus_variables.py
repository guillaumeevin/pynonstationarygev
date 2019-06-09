import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array


class CrocusSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent'
    UNIT = 'kg per m2 or mm'

    @classmethod
    def keyword(cls):
        return 'WSN_T_ISBA'


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls):
        return "DSN_T_ISBA"
