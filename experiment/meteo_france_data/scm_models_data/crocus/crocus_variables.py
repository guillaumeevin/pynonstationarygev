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
    def keyword(cls, nb_consecutive_days=3):
        return 'SWE_{}DY_ISBA'.format(nb_consecutive_days)


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    @classmethod
    def keyword(cls, nb_consecutive_days=3):
        return "SD_{}DY_ISBA".format(nb_consecutive_days)
