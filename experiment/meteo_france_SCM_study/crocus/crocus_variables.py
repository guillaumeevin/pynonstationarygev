import numpy as np

from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable


class CrocusVariable(AbstractVariable):

    def __init__(self, dataset, altitude, variable_name):
        super().__init__(dataset, altitude)
        self.variable_name = variable_name

    @property
    def daily_time_serie(self):
        time_serie_every_6_hours = np.array(self.dataset.variables[self.variable_name])[:, 0, :]
        if self.altitude == 2400:
            time_serie_daily = time_serie_every_6_hours
        else:
            nb_days = len(time_serie_every_6_hours) // 4
            # The first value of each day is selected (in order to be comparable to an instantaneous value)
            time_serie_daily = np.array([time_serie_every_6_hours[4 * i] for i in range(nb_days)])
            # Take the mean over a full day (WARNING: by doing that I am potentially destroying some maxima)
            # (I could also create a special mode where I take the maximum instead of the mean here)
            # time_serie_daily = np.array([np.mean(time_serie_every_6_hours[4 * i:4 * (i + 1)], axis=0)
            #                              for i in range(nb_days)])
        return time_serie_daily


class CrocusSweVariable(CrocusVariable):
    NAME = 'Snow Water Equivalent'

    def __init__(self, dataset, altitude):
        # Units are kg m-2
        super().__init__(dataset, altitude, 'SNOWSWE')


class CrocusDepthVariable(CrocusVariable):
    """Crocus Depth  data is every 6 hours
    To obtain daily data, we take the average over the 4 slots of 6 hours that compose a full day """
    NAME = 'Snow Depth'

    def __init__(self, dataset, altitude):
        # Units are m
        super().__init__(dataset, altitude, "SNOWDEPTH")
