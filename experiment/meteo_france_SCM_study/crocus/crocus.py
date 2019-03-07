import numpy as np

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus_variables import CrocusSweVariable, CrocusDepthVariable


class Crocus(AbstractStudy):
    """
    In the Crocus data, there is no 'massifsList' variable, thus we assume massifs are ordered just like Safran data
    """

    def __init__(self, variable_class, altitude=1800):
        assert variable_class in [CrocusSweVariable, CrocusDepthVariable]
        super().__init__(variable_class, altitude)
        self.model_name = 'Crocus'

    @property
    def variable_name(self):
        suffix = '' if self.altitude == 2400 else ' average of data observed every 6 hours'
        return super().variable_name + suffix

    def annual_aggregation_function(self):
        return np.mean


class CrocusSwe(Crocus):

    def __init__(self, altitude=1800):
        super().__init__(CrocusSweVariable, altitude)


class ExtendedCrocusSwe(AbstractExtendedStudy, CrocusSwe):
    pass


class CrocusDepth(Crocus):

    def __init__(self, altitude=1800):
        super().__init__(CrocusDepthVariable, altitude)


class ExtendedCrocusDepth(AbstractExtendedStudy, CrocusDepth):
    pass


if __name__ == '__main__':
    for variable_class in [CrocusSweVariable, CrocusDepthVariable]:
        study = Crocus(variable_class=variable_class, altitude=2400)
        d = study.year_to_dataset_ordered_dict[1960]
        time_arr = np.array(d.variables['time'])
        print(time_arr)
        # print(d)
        a = study.year_to_daily_time_serie[1960]
        print(a.shape)
