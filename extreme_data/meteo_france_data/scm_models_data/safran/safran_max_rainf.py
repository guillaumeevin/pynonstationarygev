from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranPrecipitation1Day, SafranRainfall1Day
from extreme_data.meteo_france_data.scm_models_data.studyfrommaxfiles import AbstractStudyMaxFiles
from extreme_data.meteo_france_data.scm_models_data.utils import Season


class AbstractSafranRainfallMaxFiles(AbstractStudyMaxFiles, SafranRainfall1Day):

    def __init__(self, safran_year, **kwargs):
        if 'season' in kwargs:
            season = kwargs['season']
        else:
            season = Season.annual
        if season is Season.annual:
            keyword = "max-1day-rainf-year"
        else:
            raise NotImplementedError('data not available for this season')
        super().__init__(safran_year, keyword, **kwargs)


class SafranRainfall2019(AbstractSafranRainfallMaxFiles):

    def __init__(self, **kwargs):
        super().__init__(2019, **kwargs)


if __name__ == '__main__':
    study = SafranRainfall2019(altitude=1800)
    print(study.year_to_annual_maxima[1959])
    print(len(study.column_mask))
