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


class ExtendedSafranSnowfall(AbstractExtendedStudy, SafranSnowfall):
    pass


class SafranRainfall(CumulatedStudy, Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranRainfallVariable, *args, **kwargs)


class SafranTotalPrecip(CumulatedStudy, Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranTotalPrecipVariable, *args, **kwargs)

    def load_variable_array(self, dataset):
        return [np.array(dataset.variables[k]) for k in self.load_keyword()]

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        variable_array_snowfall, variable_array_rainfall = variable_array
        return self.variable_class(variable_array_snowfall, variable_array_rainfall, self.nb_consecutive_days)


class ExtendedSafranTotalPrecip(AbstractExtendedStudy, SafranTotalPrecip):
    pass


class SafranTemperature(Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranTemperatureVariable, *args, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)


if __name__ == '__main__':
    study = SafranSnowfall()
    d = study.year_to_dataset_ordered_dict[1958]
    # print(d.variables['time'])
    # print(study.all_massif_names)
    # print(study.massif_name_to_altitudes)
    # print(study.year_to_daily_time_serie_array[1958].shape)
    # print(study.missing_massif_name)

    # print(len(d.variables['time']))
    # print(study.year_to_annual_total)
    # print(study.df_annual_total.columns)
