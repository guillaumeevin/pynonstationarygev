import numpy as np

from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.safran.safran_variable import SafranSnowfallVariable, \
    SafranPrecipitationVariable, SafranTemperatureVariable


class Safran(AbstractStudy):

    def __init__(self, variable_class: type, *args, **kwargs):
        assert variable_class in [SafranSnowfallVariable, SafranPrecipitationVariable, SafranTemperatureVariable]
        super().__init__(variable_class, *args, **kwargs)
        self.model_name = 'Safran'


class SafranFrequency(Safran):

    def __init__(self, variable_class: type, nb_consecutive_days=1, *args, **kwargs):
        super().__init__(variable_class, *args, **kwargs)
        self.nb_consecutive_days = nb_consecutive_days

    def instantiate_variable_object(self, dataset) -> AbstractVariable:
        return self.variable_class(dataset, self.nb_consecutive_days)

    @property
    def variable_name(self):
        return super().variable_name + ' cumulated over {} days'.format(self.nb_consecutive_days)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.sum(*args, **kwargs)


class SafranSnowfall(SafranFrequency):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfallVariable, *args, **kwargs)


class ExtendedSafranSnowfall(AbstractExtendedStudy, SafranSnowfall):
    pass


class SafranPrecipitation(SafranFrequency):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranPrecipitationVariable, *args, **kwargs)


class SafranTemperature(Safran):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranTemperatureVariable, *args, **kwargs)

    def annual_aggregation_function(self, *args, **kwargs):
        return np.mean(*args, **kwargs)


if __name__ == '__main__':
    study = SafranSnowfall()
    d = study.year_to_dataset_ordered_dict[1958]
    print(d.variables['time'])
    # print(study.year_to_daily_time_serie[1958].shape)
    # print(len(d.variables['time']))
    print(study.year_to_annual_total)
    print(study.df_annual_total)
