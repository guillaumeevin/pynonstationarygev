from experiment.meteo_france_SCM_study.abstract_extended_study import AbstractExtendedStudy
from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.abstract_variable import AbstractVariable
from experiment.meteo_france_SCM_study.safran.safran_snowfall_variable import SafranSnowfallVariable


class Safran(AbstractStudy):

    def __init__(self, altitude=1800, nb_days_of_snowfall=1):
        super().__init__(SafranSnowfallVariable, altitude)
        self.nb_days_of_snowfall = nb_days_of_snowfall
        self.model_name = 'Safran'

    def instantiate_variable_object(self, dataset) -> AbstractVariable:
        return self.variable_class(dataset, self.nb_days_of_snowfall)

    @property
    def variable_name(self):
        return super().variable_name + ' cumulated over {} days'.format(self.nb_days_of_snowfall)


class ExtendedSafran(AbstractExtendedStudy, Safran):
    pass


if __name__ == '__main__':
    study = Safran()
    d = study.year_to_dataset_ordered_dict[1958]
    print(d.variables['time'])
    print(study.year_to_daily_time_serie[1958].shape)
    print(len(d.variables['time']))