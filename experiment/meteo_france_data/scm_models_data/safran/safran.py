import numpy as np

from experiment.meteo_france_data.scm_models_data.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from experiment.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from experiment.meteo_france_data.scm_models_data.cumulated_study import CumulatedStudy
from experiment.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable, \
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

    def __init__(self, *args, **kwargs):
        CumulatedStudy.__init__(self, SafranSnowfallVariable, *args, **kwargs)
        Safran.__init__(self, SafranSnowfallVariable, *args, **kwargs)


class SafranSnowfall1Day(SafranSnowfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=1, *args, **kwargs)


class SafranSnowfall3Days(SafranSnowfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=3, *args, **kwargs)


class SafranSnowfall5Days(SafranSnowfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=5, *args, **kwargs)


class SafranSnowfall7Days(SafranSnowfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=7, *args, **kwargs)


class ExtendedSafranSnowfall(AbstractExtendedStudy, SafranSnowfall):
    pass


class SafranRainfall(CumulatedStudy, Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranRainfallVariable, *args, **kwargs)


class SafranRainfall1Day(SafranRainfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=1, *args, **kwargs)


class SafranRainfall3Days(SafranRainfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=3, *args, **kwargs)


class SafranRainfall5Days(SafranRainfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=5, *args, **kwargs)


class SafranRainfall7Days(SafranRainfall):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=7, *args, **kwargs)


class SafranPrecipitation(CumulatedStudy, Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranTotalPrecipVariable, *args, **kwargs)

    def load_variable_array(self, dataset):
        return [np.array(dataset.variables[k]) for k in self.load_keyword()]

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        variable_array_snowfall, variable_array_rainfall = variable_array
        return self.variable_class(variable_array_snowfall, variable_array_rainfall, self.nb_consecutive_days)


class SafranPrecipitation1Day(SafranPrecipitation):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=1, *args, **kwargs)


class SafranPrecipitation3Days(SafranPrecipitation):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=3, *args, **kwargs)


class SafranPrecipitation5Days(SafranPrecipitation):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=5, *args, **kwargs)


class SafranPrecipitation7Days(SafranPrecipitation):

    def __init__(self, *args, **kwargs):
        super().__init__(nb_consecutive_days=7, *args, **kwargs)


class ExtendedSafranPrecipitation(AbstractExtendedStudy, SafranPrecipitation):
    pass


class SafranTemperature(Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranTemperatureVariable, *args, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)


if __name__ == '__main__':
    study = SafranRainfall1Day()
    print(study.year_to_annual_maxima[1959])
    print(study.ordered_years)
