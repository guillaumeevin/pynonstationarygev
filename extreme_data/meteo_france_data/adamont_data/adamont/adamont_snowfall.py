from extreme_data.meteo_france_data.adamont_data.abstract_adamont_study import AbstractAdamontStudy
from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import \
    SafranSnowfallSimulationVariable
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import AdamontScenario
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class AdamontSnowfall(AbstractAdamontStudy):

    def __init__(self, altitude: int = 1800,
                 year_min=None, year_max=None,
                 multiprocessing=True, season=Season.annual,
                 french_region=FrenchRegion.alps,
                 scenario=AdamontScenario.histo, gcm_rcm_couple=('CNRM-CM5', 'ALADIN53'),
                 adamont_version=2):
        super().__init__(SafranSnowfallSimulationVariable, altitude,
                         year_min, year_max,
                         multiprocessing, season, french_region, scenario, gcm_rcm_couple, adamont_version)


if __name__ == '__main__':
    study = AdamontSnowfall(altitude=1800, adamont_version=1, gcm_rcm_couple=('HadGEM2-ES', 'RACMO22E'),
                            scenario=AdamontScenario.rcp85_extended)
    pass
    print(study.year_to_annual_maxima)
    print(study.year_to_annual_maxima)
