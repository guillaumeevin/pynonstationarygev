import numpy as np

from extreme_data.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.safran.cumulated_study import CumulatedStudy
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable, \
    SafranRainfallVariable, SafranTemperatureVariable, SafranTotalPrecipVariable


class Safran(AbstractStudy):

    def __init__(self, variable_class: type, *args, **kwargs):
        assert variable_class in [SafranSnowfallVariable, SafranRainfallVariable, SafranTemperatureVariable,
                                  SafranTotalPrecipVariable]
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
    altitude = 900
    year_min = 1959
    year_max = 2000
    study = SafranRainfall1Day(altitude, year_min=year_min, year_max=year_max)
    d = study.year_to_dataset_ordered_dict[1959]
    print(d.keywords)
    print(d.variables.keys())
    print(study.year_to_annual_maxima[1959])
    print(study.ordered_years)
