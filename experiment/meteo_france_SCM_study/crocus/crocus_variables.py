import numpy as np

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):
    """Crocus data is every 6 hours. To obtain daily data, we select one data out of 4
    (in order to have data that will still be comparable to an instantaneous variable"""

    def __init__(self, dataset, altitude, variable_name):
        super().__init__(dataset, altitude)
        self.variable_name = variable_name

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return np.array(self.dataset.variables[self.variable_name])


class CrocusSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent'
    UNIT = 'kg per m2 or mm'

    def __init__(self, dataset, altitude):
        super().__init__(dataset, altitude, 'SWE_1DY_ISBA')


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'
    UNIT = 'm'

    def __init__(self, dataset, altitude):
        super().__init__(dataset, altitude, "SD_1DY_ISBA")

