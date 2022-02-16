from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.crocus.crocus import CrocusSnowLoadTotal, CrocusSweTotal
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import AbstractSnowLoadVariable
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles


class AbstractCrocusSweMaxFiles(AbstractStudyMaxFiles, CrocusSweTotal):

    def __init__(self, safran_year, **kwargs):
        super().__init__(safran_year, "swe-max-year-NN", **kwargs)


class AbstractCrocusSnowLoadMaxFiles(AbstractStudyMaxFiles, CrocusSnowLoadTotal):

    def __init__(self, safran_year, **kwargs):
        super().__init__(safran_year, "swe-max-year-NN", **kwargs)

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        year_to_swe_annual_maxima = super().year_to_annual_maxima
        year_to_annual_maxima = OrderedDict()
        for year, swe_annual_maxima in year_to_swe_annual_maxima.items():
            snow_load_annual_maxima = AbstractSnowLoadVariable.transform_swe_into_snow_load(swe_annual_maxima)
            year_to_annual_maxima[year] = snow_load_annual_maxima
        return year_to_annual_maxima


class CrocusSnowLoad2020(AbstractCrocusSnowLoadMaxFiles):
    YEAR_MAX = 2020

    def __init__(self, **kwargs):
        if ('year_max' not in kwargs) or (kwargs['year_max'] is None):
            kwargs['year_max'] = self.YEAR_MAX
        super().__init__(2020, **kwargs)


class CrocusSnowLoad2019(AbstractCrocusSnowLoadMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)


if __name__ == '__main__':
    study = CrocusSnowLoad2019(altitude=1800)
    print(len(study.column_mask))
