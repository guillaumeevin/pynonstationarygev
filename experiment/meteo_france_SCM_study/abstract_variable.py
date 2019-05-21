import numpy as np


class AbstractVariable(object):
    """
    All Variable (CROCUS & SAFRAN) are available since 1958-08-01 06:00:00
    """

    NAME = ''
    UNIT = ''

    def __init__(self, variable_array):
        self.variable_array = variable_array

    @classmethod
    def keyword(cls):
        raise NotImplementedError

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        # Return an array of size length of time series x nb_massif
        raise NotImplementedError
