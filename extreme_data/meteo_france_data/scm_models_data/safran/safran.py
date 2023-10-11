from collections import OrderedDict
from typing import List

import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.safran.cumulated_study import CumulatedStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable, \
    SafranRainfallVariable, SafranTemperatureVariable, SafranTotalPrecipVariable, \
    SafranNormalizedPrecipitationRateOnWetDaysVariable, SafranNormalizedPrecipitationRateVariable, \
    SafranDateFirstSnowfallVariable


class Safran(AbstractStudy):
    SAFRAN_VARIABLES = [SafranSnowfallVariable,
                        SafranRainfallVariable,
                        SafranTemperatureVariable,
                        SafranTotalPrecipVariable,
                        SafranNormalizedPrecipitationRateVariable,
                        SafranNormalizedPrecipitationRateOnWetDaysVariable,
                        SafranDateFirstSnowfallVariable]

    def __init__(self, variable_class: type, *args, **kwargs):
        assert variable_class in self.SAFRAN_VARIABLES
        super().__init__(variable_class, *args, **kwargs)
        self.model_name = 'Safran'

    def annual_aggregation_function(self, *args, **kwargs):
        return np.sum(*args, **kwargs)


class SafranSnowfall(Safran, CumulatedStudy):

    def __init__(self, **kwargs):
        super().__init__(SafranSnowfallVariable, **kwargs)


class SafranSnowfall1Day(SafranSnowfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=1, **kwargs)

class SafranDateFirstSnowfall(Safran, CumulatedStudy):

    def __init__(self, **kwargs):
        super().__init__(SafranDateFirstSnowfallVariable, nb_consecutive_days=1, **kwargs)
        self.massif_id_to_remove = set()
        for year in self.ordered_years:
            s = super().daily_time_series(year)
            column_has_nan = np.isnan(s).any(axis=0)
            index_with_nan = list(np.nonzero(column_has_nan)[0])
            if len(index_with_nan) > 0:
                self.massif_id_to_remove.update(set(index_with_nan))
        self.massif_id_to_keep = tuple([i for i in range(s.shape[1])
                                        if i not in self.massif_id_to_remove])

    def daily_time_series(self, year):
        return super().daily_time_series(year)[:, self.massif_id_to_keep]

    @property
    def study_massif_names(self) -> List[str]:
        return [m for i, m in enumerate(super().study_massif_names) if i not in self.massif_id_to_remove]


class SafranSnowfall3Days(SafranSnowfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=3, **kwargs)


class SafranSnowfall5Days(SafranSnowfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=5, **kwargs)


class SafranSnowfall7Days(SafranSnowfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=7, **kwargs)


class ExtendedSafranSnowfall(AbstractExtendedStudy, SafranSnowfall):
    pass


class SafranRainfall(CumulatedStudy, Safran):

    def __init__(self, **kwargs):
        super().__init__(SafranRainfallVariable, **kwargs)


class SafranRainfall1Day(SafranRainfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=1, **kwargs)


class SafranRainfall3Days(SafranRainfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=3, **kwargs)


class SafranRainfall5Days(SafranRainfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=5, **kwargs)


class SafranRainfall7Days(SafranRainfall):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=7, **kwargs)


class SafranNormalizedPreciptationRate(CumulatedStudy, Safran):

    def __init__(self, **kwargs):
        super().__init__(SafranNormalizedPrecipitationRateVariable, **kwargs)

    def load_variable_array(self, dataset):
        return [np.array(dataset.variables[k]) for k in self.load_keyword()]

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        variable_array_temperature, variable_array_snowfall, variable_array_rainfall = variable_array
        return self.variable_class(variable_array_temperature,
                                   variable_array_snowfall, variable_array_rainfall, self.nb_consecutive_days)


class SafranNormalizedPreciptationRateOnWetDays(CumulatedStudy, Safran):

    def __init__(self, **kwargs):
        super().__init__(SafranNormalizedPrecipitationRateOnWetDaysVariable, **kwargs)

    def load_variable_array(self, dataset):
        return [np.array(dataset.variables[k]) for k in self.load_keyword()]

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        variable_array_temperature, variable_array_snowfall, variable_array_rainfall = variable_array
        return self.variable_class(variable_array_temperature,
                                   variable_array_snowfall, variable_array_rainfall, self.nb_consecutive_days)

    @property
    def _year_to_daily_time_serie_array(self) -> OrderedDict:
        # Filter and keep only values different than np.nan
        year_to_time_series = super()._year_to_daily_time_serie_array
        updated_year_to_time_series = OrderedDict()
        for year, time_serie in year_to_time_series.items():
            time_serie_without_nan = time_serie[~np.isnan(time_serie)]
            assert not np.isnan(time_serie_without_nan).any()
            updated_year_to_time_series[year] = time_serie_without_nan
        return updated_year_to_time_series


class SafranPrecipitation(CumulatedStudy, Safran):

    def __init__(self, **kwargs):
        super().__init__(SafranTotalPrecipVariable, **kwargs)

    def load_variable_array(self, dataset):
        return [np.array(dataset.variables[k]) for k in self.load_keyword()]

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        variable_array_snowfall, variable_array_rainfall = variable_array
        return self.variable_class(variable_array_snowfall, variable_array_rainfall, self.nb_consecutive_days)


class SafranPrecipitation1Day(SafranPrecipitation):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=1, **kwargs)


class SafranPrecipitation3Days(SafranPrecipitation):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=3, **kwargs)


class SafranPrecipitation5Days(SafranPrecipitation):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=5, **kwargs)


class SafranPrecipitation7Days(SafranPrecipitation):

    def __init__(self, **kwargs):
        super().__init__(nb_consecutive_days=7, **kwargs)


class ExtendedSafranPrecipitation(AbstractExtendedStudy, SafranPrecipitation):
    pass


class SafranTemperature(Safran):

    def __init__(self, **kwargs):
        super().__init__(SafranTemperatureVariable, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)


if __name__ == '__main__':
    # altitude = 1800
    altitude = 900
    year_min = 1959
    year_max = 2019
    # study = SafranSnowfall(altitude=altitude, year_min=year_min, year_max=year_max)
    # print(study.year_to_daily_time_serie_array[1959].shape)
    # print(study.massif_name_to_daily_time_series['Vanoise'].shape)
