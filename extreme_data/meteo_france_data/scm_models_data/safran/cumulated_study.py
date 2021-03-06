from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable

NB_DAYS = [1, 3, 5, 7]


class CumulatedStudy(AbstractStudy):
    def __init__(self, variable_class: type, nb_consecutive_days: int = 3, *args, **kwargs):
        assert nb_consecutive_days in NB_DAYS, nb_consecutive_days
        super().__init__(variable_class, *args, **kwargs)
        self.nb_consecutive_days = nb_consecutive_days

    def instantiate_variable_object(self, variable_array) -> AbstractVariable:
        return self.variable_class(variable_array, self.nb_consecutive_days)

    @property
    def variable_name(self):
        return super().variable_name + ' {} day(s)'.format(self.nb_consecutive_days)
