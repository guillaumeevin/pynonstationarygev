from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, SafranSnowfall3Days
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles


class AbstractSafranSnowfallMaxFiles(AbstractStudyMaxFiles, SafranSnowfall1Day):

    def __init__(self, safran_year, **kwargs):
        super().__init__(safran_year, self._indicator_name(), **kwargs)


    @classmethod
    def _indicator_name(cls):
        return "max-1day-snowf"

class SafranSnowfall2020(AbstractSafranSnowfallMaxFiles):
    YEAR_MAX = 2020

    def __init__(self, **kwargs):
        if ('year_max' not in kwargs) or (kwargs['year_max'] is None):
            kwargs['year_max'] = self.YEAR_MAX
        super().__init__(2020, **kwargs)


class SafranSnowfall2019(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)

class SafranSnowfall2022(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2022, **kwargs)

    @classmethod
    def _indicator_name(cls):
        return "max-1day-snowf-year"

class SafranSnowfall3Days2022(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2022, **kwargs)

    @classmethod
    def _indicator_name(cls):
        return "max-3day-consec-snowf-year"

class SafranSnowfall5Days2022(AbstractSafranSnowfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2022, **kwargs)

    @classmethod
    def _indicator_name(cls):
        return "max-5day-consec-snowf-year"

if __name__ == '__main__':
    study = SafranSnowfall2019(altitude=1800)
    print(study.year_to_annual_maxima[2000])
    study = SafranSnowfall2022(altitude=1800)
    print(study.year_to_annual_maxima[2000])
    study = SafranSnowfall3Days2022(altitude=1800)
    print(study.year_to_annual_maxima[2000])