import numpy as np

from extreme_data.meteo_france_data.adamont_data.abstract_simulation_study import SimulationStudy
from extreme_data.meteo_france_data.scm_models_data.abstract_study import YEAR_MIN, YEAR_MAX
from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class SafranSnowfallSimulationVariable(AbstractVariable):

    @property
    def daily_time_serie_array(self) -> np.ndarray:
        return self.variable_array

    @classmethod
    def keyword(cls):
        return 'SNOW'


class SafranSnowfallSimulationRCP85(SimulationStudy):

    def __init__(self, altitude: int = 1800, year_min=YEAR_MIN, year_max=YEAR_MAX,
                 multiprocessing=True, orientation=None, slope=20.0, season=Season.annual,
                 french_region=FrenchRegion.alps, split_years=None):
        super().__init__(SafranSnowfallSimulationVariable, altitude, year_min, year_max, multiprocessing, orientation,
                         slope,
                         season, french_region, split_years, "RCP85")


if __name__ == '__main__':
    study = SafranSnowfallSimulationRCP85(altitude=1800)
    print(study.year_to_annual_maxima)
