import numpy as np

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus_variables import CrocusSweVariable, CrocusDepthVariable


class Crocus(AbstractStudy):
    """
    In the Crocus data, there is no 'massifsList' variable, thus we assume massifs are ordered just like Safran data
    """

    def __init__(self, variable_class, *args, **kwargs):
        assert variable_class in [CrocusSweVariable, CrocusDepthVariable]
        super().__init__(variable_class, *args, **kwargs)
        self.model_name = 'Crocus'

    @property
    def variable_name(self):
        suffix = '' if self.altitude == 2400 else ' instantaneous data observed sampled every 24 hours'
        return super().variable_name + suffix

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        # In the Durand paper, we only want the data from November to April
        # 91 = 30 + 31 + 30 first days of the time serie correspond to the month of August + September + October
        # 92 = 31 + 30 + 31 last days correspond to the month of May + June + JUly
        return super().apply_annual_aggregation(time_serie[91:-92, ...])


class CrocusSwe(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusSweVariable, *args, **kwargs)


class ExtendedCrocusSwe(AbstractExtendedStudy, CrocusSwe):
    pass


class CrocusDepth(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)


class ExtendedCrocusDepth(AbstractExtendedStudy, CrocusDepth):
    pass


if __name__ == '__main__':
    for variable_clas in [CrocusSweVariable, CrocusDepthVariable]:
        study = Crocus(variable_class=variable_clas, altitude=2400)
        d = study.year_to_dataset_ordered_dict[1960]
        time_arr = np.array(d.variables['time'])
        print(time_arr)
        # print(d)
        a = study.year_to_daily_time_serie_array[1960]
        print(a.shape)
