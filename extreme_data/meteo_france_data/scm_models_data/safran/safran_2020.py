from collections import OrderedDict

import numpy as np
from cached_property import cached_property
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day


class SnowfallSafran2020(SafranSnowfall1Day):
    nc_filepath = """/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/SAFRAN_montagne_CROCUS_2020/max-1day-snowf_SAFRAN.nc"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def ordered_years(self):
        return list(range(1959, 2021))

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        year_to_annual_maxima = OrderedDict()
        dataset = Dataset(SnowfallSafran2020.nc_filepath)
        annual_maxima = np.array(dataset.variables['max-1day-snowf'])
        assert annual_maxima.shape[1] == len(self.column_mask)
        annual_maxima = annual_maxima[:, self.column_mask]
        for year, a in zip(self.ordered_years, annual_maxima):
            year_to_annual_maxima[year] = a
        return year_to_annual_maxima



if __name__ == '__main__':
    test_safran_2020_loader()
