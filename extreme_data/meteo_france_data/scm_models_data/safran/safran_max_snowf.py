from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles


class AbstractSafranSnowfallMaxFiles(AbstractStudyMaxFiles, SafranSnowfall1Day):

    def __init__(self, safran_year, **kwargs):
        super().__init__(safran_year, "max-1day-snowf", **kwargs)


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
