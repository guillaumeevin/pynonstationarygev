from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import \
    SafranSnowfallSimulationVariable, SafranPrecipitationSimulationVariable, SafranSnowfall3daysSimulationVariable, \
    SafranSnowfall5daysSimulationVariable
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class AdamontSnowfall(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfallSimulationVariable, *args, **kwargs)

class AdamontSnowfall3days(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfall3daysSimulationVariable, *args, **kwargs)


class AdamontSnowfall5days(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranSnowfall5daysSimulationVariable, *args, **kwargs)

class AdamontPrecipitation(AbstractAdamontStudy):

    def __init__(self, *args, **kwargs):
        super().__init__(SafranPrecipitationSimulationVariable, *args, **kwargs)



if __name__ == '__main__':
    study = AdamontSnowfall3days(altitude=1800, gcm_rcm_couple=('HadGEM2-ES', 'RACMO22E'),
                            scenario=AdamontScenario.rcp85_extended)
    print(study.year_to_annual_maxima)
