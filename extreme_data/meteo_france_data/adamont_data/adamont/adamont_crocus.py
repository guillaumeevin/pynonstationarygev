from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import CrocusSweSimulationVariable, \
    CrocusTotalSnowLoadVariable
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusVariable
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class AdamontSwe(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusSweSimulationVariable, *args, **kwargs)


class AdamontSnowLoad(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(CrocusTotalSnowLoadVariable, *args, **kwargs)

    def load_annual_maxima(self, dataset):
        return CrocusVariable.snow_load_multiplication_factor * super().load_annual_maxima(dataset)


if __name__ == '__main__':
    for study_class in [AdamontSwe, AdamontSnowLoad][1:]:
        study = study_class(altitude=1800, gcm_rcm_couple=('HadGEM2-ES', 'RACMO22E'),
                                scenario=AdamontScenario.rcp85_extended)
        print(study.year_to_annual_maxima[2000])
        print(study.year_to_annual_maxima_index[2000])