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
        return super().variable_name() + 'cumulated over {} days'.format(self.nb_days_of_snowfall)

