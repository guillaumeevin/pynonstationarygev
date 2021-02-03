from collections import OrderedDict

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.safran.safran import SafranSnowfall1Day, \
    SafranSnowfallCenterOnDay1day
from extreme_data.meteo_france_data.scm_models_data.safran.safran_max_snowf import SafranSnowfall2020, \
    SafranSnowfall2019


class AbstractGapBetweenTwoStudyClass(SafranSnowfall1Day):

    def __init__(self, study_class1, study_class2, **kwargs):
        super().__init__(**kwargs)
        self.study_1 = study_class1(**kwargs)
        self.study_2 = study_class2(**kwargs)

    @property
    def ordered_years(self):
        return [i for i in list(range(1959, 2020)) if self.year_min <= i <= self.year_max]

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        year_to_annual_maxima = OrderedDict()
        for year in self.ordered_years:
            annual_maxima_1 = self.study_1.year_to_annual_maxima[year]
            annual_maxima_2 = self.study_2.year_to_annual_maxima[year]
            year_to_annual_maxima[year] = annual_maxima_2 - annual_maxima_1
        return year_to_annual_maxima


class GapBetweenSafranSnowfall2019And2020(AbstractGapBetweenTwoStudyClass):

    def __init__(self, **kwargs):
        super().__init__(SafranSnowfall2019, SafranSnowfall2020, **kwargs)


class GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered(AbstractGapBetweenTwoStudyClass):

    def __init__(self, **kwargs):
        super().__init__(SafranSnowfall2019, SafranSnowfallCenterOnDay1day, **kwargs)


class GapBetweenSafranSnowfall2019AndMySafranSnowfall2019(AbstractGapBetweenTwoStudyClass):

    def __init__(self, **kwargs):
        super().__init__(SafranSnowfall2019, SafranSnowfall1Day, **kwargs)


if __name__ == '__main__':
    study = GapBetweenSafranSnowfall2019AndMySafranSnowfall2019Recentered(altitude=1800)
    study.year_to_annual_maxima[1960]
