from experiment.meteo_france_SCM_study.abstract_study import AbstractStudy
from experiment.meteo_france_SCM_study.crocus.crocus_variables import CrocusSweVariable, CrocusDepthVariable


class Crocus(AbstractStudy):

    def __init__(self, safran_altitude=1800, crocus_variable_class=CrocusSweVariable):
        super().__init__(safran_altitude)
        self.model_name = 'Crocus'
        assert crocus_variable_class in [CrocusSweVariable, CrocusDepthVariable]
        self.variable_class = crocus_variable_class

if __name__ == '__main__':
    for variable_class in [CrocusSweVariable, CrocusDepthVariable]:
        study = Crocus(crocus_variable_class=variable_class)
        # d = study.year_to_dataset_ordered_dict[1960]
        # print(d)
        a = study.year_to_daily_time_serie[1960]
        print(a.shape)