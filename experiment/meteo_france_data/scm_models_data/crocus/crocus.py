from collections import OrderedDict

import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusTotalSweVariable, \
    CrocusDepthVariable, CrocusRecentSweVariable, TotalSnowLoadVariable, RecentSnowLoadVariable, \
    CrocusSnowLoadEurocodeVariable, CrocusDensityVariable


class Crocus(AbstractStudy):
    """
    In the Crocus data, there is no 'massifsList' variable, thus we assume massifs are ordered just like Safran data
    """

    def __init__(self, variable_class, *args, **kwargs):
        assert variable_class in [CrocusTotalSweVariable, CrocusDepthVariable, CrocusRecentSweVariable,
                                  RecentSnowLoadVariable, TotalSnowLoadVariable, CrocusSnowLoadEurocodeVariable,
                                  CrocusDensityVariable]
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


# Create some class that enables to deal directly with the snow load


class CrocusSnowLoadTotal(Crocus):
    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, TotalSnowLoadVariable, *args, **kwargs)


class CrocusSnowLoad3Days(CrocusSweTotal):
    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, RecentSnowLoadVariable, *args, **kwargs)


class ExtendedCrocusSweTotal(AbstractExtendedStudy, CrocusSweTotal):
    pass


class CrocusDepth(Crocus):

    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, CrocusDepthVariable, *args, **kwargs)

    def apply_annual_aggregation(self, time_serie):
        return self.winter_annual_aggregation(time_serie)


class CrocusSnowLoadEurocode(Crocus):

    def __init__(self, *args, **kwargs):
        Crocus.__init__(self, CrocusSnowLoadEurocodeVariable, *args, **kwargs)


class ExtendedCrocusDepth(AbstractExtendedStudy, CrocusDepth):
    pass


class CrocusDaysWithSnowOnGround(Crocus):
    """Having snow on the ground is equivalent to snow depth > 0"""

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDepthVariable, *args, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.count_nonzero(*args, **kwargs)


if __name__ == '__main__':
    for study in [CrocusSwe3Days(altitude=900, orientation=90.0)]:
        d = study.year_to_dataset_ordered_dict[1958]
        print(d)
        print(study.reanalysis_path)
        for v in ['aspect', 'slope', 'ZS', 'massif_num']:
            a = np.array(d[v])
            print(list(a))
            print(sorted(list(set(a))))
        print(study.year_to_daily_time_serie_array[1958])
    study = CrocusSnowLoadTotal(altitude=900)
    print(study.year_to_annual_maxima_index)
    print(study.year_to_daily_time_serie_array)

    # a = study.year_to_daily_time_serie_array[1960]
    # print(a)
