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

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)

    def winter_annual_aggregation(self, time_serie):
        # In the Durand paper, we only want the data from November to April
        # 91 = 30 + 31 + 30 first days of the time serie correspond to the month of August + September + October
        # 92 = 31 + 30 + 31 last days correspond to the month of May + June + JUly
        return super().apply_annual_aggregation(time_serie[91:-92, ...])


class CrocusSwe(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusSweVariable, *args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        return self.winter_annual_aggregation(time_serie)


class ExtendedCrocusSwe(AbstractExtendedStudy, CrocusSwe):
    pass


class CrocusDepth(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        return self.winter_annual_aggregation(time_serie)


class ExtendedCrocusDepth(AbstractExtendedStudy, CrocusDepth):
    pass


class CrocusDaysWithSnowOnGround(Crocus):
    """Having snow on the ground is equivalent to snow depth > 0"""

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.count_nonzero(*args, **kwargs)


if __name__ == '__main__':
    for variable_class in [CrocusSweVariable, CrocusDepthVariable][:1]:
        study = Crocus(variable_class=variable_class, altitude=2400)
        d = study.year_to_dataset_ordered_dict[1960]
        print(study.df_massifs_longitude_and_latitude)
        time_arr = np.array(d.variables['time'])
        a = study.year_to_daily_time_serie_array[1960]
        print(a.shape)
