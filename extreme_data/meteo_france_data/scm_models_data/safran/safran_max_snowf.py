from collections import OrderedDict
import os.path as op

import numpy as np
from cached_property import cached_property
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, Safran
from extreme_data.utils import DATA_PATH


class AbstractSafranSnowfallMaxFiles(SafranSnowfall1Day):

    def __init__(self, safran_year, **kwargs):
        super().__init__(**kwargs)
        self.nc_filepath = op.join(DATA_PATH, 'SAFRAN_montagne-CROCUS_{}'.format(safran_year),
                                   'max-1day-snowf_SAFRAN.nc')
        self.safran_year = safran_year

    @property
    def ordered_years(self):
        return [i for i in self.all_years if self.year_min <= i <= self.year_max]

    @property
    def all_years(self):
        return list(range(1959, self.safran_year + 1))

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        year_to_annual_maxima = OrderedDict()
        dataset = Dataset(self.nc_filepath)
        annual_maxima = np.array(dataset.variables['max-1day-snowf'])
        assert annual_maxima.shape[1] == len(self.column_mask)
        annual_maxima = annual_maxima[:, self.column_mask]
        for year, maxima in zip(self.all_years, annual_maxima):
            if self.year_min <= year <= self.year_max:
                year_to_annual_maxima[year] = maxima
        return year_to_annual_maxima

    @property
    def variable_name(self):
        return self.variable_class.NAME + str(self.safran_year) + ' ({})'.format(self.variable_unit)


class SafranSnowfall2020(AbstractSafranSnowfallMaxFiles):
    YEAR_MAX = 2020

    def __init__(self, **kwargs):
        if ('year_max' not in kwargs) or (kwargs['year_max'] is None):
            kwargs['year_max'] = self.YEAR_MAX
        super().__init__(2020, **kwargs)


class SafranSnowfall2019(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)


if __name__ == '__main__':
    study = SafranSnowfall2020(altitude=1800)
    print(len(study.column_mask))
