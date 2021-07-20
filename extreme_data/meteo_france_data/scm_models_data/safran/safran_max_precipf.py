from collections import OrderedDict
import os.path as op

import numpy as np
from cached_property import cached_property
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, Safran, \
    SafranPrecipitation1Day
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles
from extreme_data.meteo_france_data.scm_models_data.utils import Season
from extreme_data.utils import DATA_PATH


class AbstractSafranPrecipitationMaxFiles(AbstractStudyMaxFiles, SafranPrecipitation1Day):

    def __init__(self, safran_year, **kwargs):
        if 'season' in kwargs:
            season = kwargs['season']
        else:
            season = Season.annual
        if season is Season.annual:
            keyword = "max-1day-precipf-year"
        elif season is Season.winter:
            keyword = "max-1day-precipf-winter-12-02"
        else:
            raise NotImplementedError('data not available for this season')
        super().__init__(safran_year, keyword, **kwargs)


class SafranPrecipitation2020(AbstractSafranPrecipitationMaxFiles):
    YEAR_MAX = 2020

    def __init__(self, **kwargs):
        if ('year_max' not in kwargs) or (kwargs['year_max'] is None):
            kwargs['year_max'] = self.YEAR_MAX
        super().__init__(2020, **kwargs)


class SafranPrecipitation2019(AbstractSafranPrecipitationMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)


if __name__ == '__main__':
    study = SafranPrecipitation2019(altitude=1800, season=Season.winter)
    print(study.year_to_annual_maxima[1959])
    print(len(study.column_mask))
