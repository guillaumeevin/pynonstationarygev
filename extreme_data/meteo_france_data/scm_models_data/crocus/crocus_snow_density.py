from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import Crocus, CrocusSweTotal
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusDensityVariable


class CrocusSnowDensity(Crocus):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusDensityVariable, *args, **kwargs)
        study_swe = CrocusSweTotal(*args, **kwargs)
        self._year_to_annual_maxima = OrderedDict()
        for year in study_swe.ordered_years:
            daily_time_series_swe = study_swe.year_to_daily_time_serie_array[year]
            daily_time_series_snow_depth = self.year_to_daily_time_serie_array[year]
            daily_time_series_density = daily_time_series_swe / daily_time_series_snow_depth
            ind_to_exclude = daily_time_series_snow_depth < 0.1
            daily_time_series_density[ind_to_exclude] = 0
            self._year_to_annual_maxima[year] = daily_time_series_density.max(axis=0)

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        return self._year_to_annual_maxima


if __name__ == '__main__':
    study = CrocusSnowDensity()
    print(study.year_to_annual_maxima[1959])
