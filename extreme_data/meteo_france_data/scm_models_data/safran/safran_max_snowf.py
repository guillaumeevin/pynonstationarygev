from collections import OrderedDict
import os.path as op

import numpy as np
from cached_property import cached_property
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, Safran


class AbstractSafranSnowfallMaxFiles(SafranSnowfall1Day):
    path = """/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets"""

    def __init__(self, safran_year, **kwargs):
        super().__init__(**kwargs)
        self.year_max = max(safran_year, self.year_max)
        self.nc_filepath = op.join(self.path, 'SAFRAN_montagne-CROCUS_{}'.format(safran_year),
                                   'max-1day-snowf_SAFRAN.nc')
        self.safran_year = safran_year

    @property
    def ordered_years(self):
        return [i for i in list(range(1959, self.safran_year+1)) if self.year_min <= i <= self.year_max]

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        year_to_annual_maxima = OrderedDict()
        dataset = Dataset(self.nc_filepath)
        annual_maxima = np.array(dataset.variables['max-1day-snowf'])
        assert annual_maxima.shape[1] == len(self.column_mask)
        annual_maxima = annual_maxima[:, self.column_mask]
        for year, a in zip(self.ordered_years, annual_maxima):
            year_to_annual_maxima[year] = a
        return year_to_annual_maxima


class SafranSnowfall2020(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2020, **kwargs)


class SafranSnowfall2019(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)
