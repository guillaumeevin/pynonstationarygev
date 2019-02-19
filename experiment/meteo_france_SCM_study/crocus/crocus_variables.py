import numpy as np

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):

    def __init__(self, dataset, variable_name):
        super().__init__(dataset)
        self.variable_name = variable_name

    @property
    def daily_time_serie(self):
        # So far the dimension of the time serie is 1460 x 23
        return np.array(self.dataset.variables[self.variable_name])[:, 0, :]


class CrocusSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent'

    def __init__(self, dataset):
        super().__init__(dataset, 'SNOWSWE')


class CrocusDepthVariable(CrocusVariable):
    NAME = 'Snow Depth'

    def __init__(self, dataset):
        super().__init__(dataset, "SNOWDEPTH")
