import numpy as np


class AbstractVariable(object):

    NAME = ''

    def __init__(self, dataset, altitude):
        self.dataset = dataset
        self.altitude = altitude

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        # Return an array of size length of time series x nb_massif
        raise NotImplementedError