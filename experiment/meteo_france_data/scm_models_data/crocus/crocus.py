import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusTotalSweVariable, \
    CrocusDepthVariable, CrocusRecentSweVariable


class Crocus(AbstractStudy):
    """
    In the Crocus data, there is no 'massifsList' variable, thus we assume massifs are ordered just like Safran data
    """

    def __init__(self, variable_class, *args, **kwargs):
        assert variable_class in [CrocusTotalSweVariable, CrocusDepthVariable, CrocusRecentSweVariable]
        super().__init__(variable_class, *args, **kwargs)
        self.model_name = 'Crocus'

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)

    def winter_annual_aggregation(self, time_serie):
        # In the Durand paper, we only want the data from November to April
        # 91 = 30 + 31 + 30 first days of the time serie correspond to the month of August + September + October
        # 92 = 31 + 30 + 31 last days correspond to the month of May + June + JUly
        return super().apply_annual_aggregation(time_serie[91:-92, ...])


class CrocusSwe3Days(Crocus):

    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, CrocusRecentSweVariable, *args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        return self.winter_annual_aggregation(time_serie)


class CrocusSweTotal(Crocus):

    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, CrocusTotalSweVariable, *args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        return self.winter_annual_aggregation(time_serie)


class ExtendedCrocusSweTotal(AbstractExtendedStudy, CrocusSweTotal):
    pass


class CrocusDepth(Crocus):

    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, CrocusDepthVariable, *args, **kwargs)

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
    for study in [CrocusSwe3Days(altitude=900)]:
        a = study.year_to_daily_time_serie_array[1960]
        print(a)
