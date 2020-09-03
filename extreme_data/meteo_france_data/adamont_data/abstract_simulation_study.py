import os.path as op
import os
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List

import numpy as np
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.scm_models_data.abstract_variable import AbstractVariable
from extreme_data.meteo_france_data.scm_models_data.crocus.crocus_variables import CrocusTotalSweVariable
from root_utils import classproperty

ADAMONT_PATH = r"/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/ADAMONT"

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy, YEAR_MIN, YEAR_MAX
from extreme_data.meteo_france_data.scm_models_data.safran.safran_variable import SafranSnowfallVariable
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion


class SimulationStudy(AbstractStudy):
    scenarios = ['HISTO', 'RCP26', 'RCP45', 'RCP85']

    def __init__(self, variable_class: type, altitude: int = 1800, year_min=YEAR_MIN, year_max=YEAR_MAX,
                 multiprocessing=True, orientation=None, slope=20.0, season=Season.annual,
                 french_region=FrenchRegion.alps, split_years=None,
                 scenario="HISTO", ensemble_idx=0):
        super().__init__(variable_class, altitude, year_min, year_max, multiprocessing, orientation, slope, season,
                         french_region, split_years)
        assert scenario in self.scenarios
        assert 0 <= ensemble_idx <= 13
        self.scenario = scenario
        self.ensemble_idx = ensemble_idx
        # Assert the massif_name are in the same order
        for i, massif_name in enumerate(self.all_massif_names()):
            assert massif_name == self.massif_number_to_massif_name[i + 1]

    @property
    def simulations_path(self):
        return op.join(ADAMONT_PATH, self.parameter, self.scenario)

    @property
    def parameter(self):
        return self.variable_class_to_parameter[self.variable_class]

    @classproperty
    def variable_class_to_parameter(cls):
        return {
            SafranSnowfallSimulationVariable: 'Snow',
            CrocusTotalSweVariable: 'SNOWSWE',
        }

    @property
    def nc_path(self):
        nc_file = os.listdir(self.simulations_path)[self.ensemble_idx]
        return op.join(self.simulations_path, nc_file)

    @property
    def massif_number_to_massif_name(self):
        # from adamont_data metadata
        s = """1	Chablais
        2	Aravis
        3	Mont-Blanc
        4	Bauges
        5	Beaufortain
        6	Haute-Tarentaise
        7	Chartreuse
        8	Belledonne
        9	Maurienne
        10	Vanoise
        11	Haute-Maurienne
        12	Grandes-Rousses
        13	Thabor
        14	Vercors
        15	Oisans
        16	Pelvoux
        17	Queyras
        18	Devoluy
        19	Champsaur
        20	Parpaillon
        21	Ubaye
        22	Haut_Var-Haut_Verdon
        23	Mercantour"""
        l = s.split('\n')
        return {int(k): m for k, m in dict([e.split() for e in l]).items()}

    @property
    def dataset(self):
        return Dataset(self.nc_path)

    @cached_property
    def ordered_years(self):
        return sorted(list(set(self.winter_year_for_each_time_step)))

    @cached_property
    def winter_year_for_each_time_step(self):
        start = datetime(year=2005, month=8, day=1, hour=6, minute=0, second=0)
        hours_after_start = np.array(self.dataset.variables['TIME'])
        dates = [start + timedelta(hours=h) for h in hours_after_start]
        winter_year = [date.year if date.month < 8 else date.year + 1 for date in dates]
        winter_year[-1] = winter_year[-2]
        return np.array(winter_year)

    @cached_property
    def year_to_variable_object(self) -> OrderedDict:
        year_to_data_list = {}
        for year in self.ordered_years:
            year_to_data_list[year] = []
        data_list = self.dataset.variables[self.variable_class.keyword]
        data_year_list = self.winter_year_for_each_time_step
        assert len(data_list) == len(data_year_list)
        for year_data, data in zip(data_year_list, data_list):
            year_to_data_list[year_data].append(data)
        year_to_variable_object = OrderedDict()
        for year in self.ordered_years:
            variable_array = np.array(year_to_data_list[year])
            year_to_variable_object[year] = self.variable_class(variable_array)
        return year_to_variable_object

    @cached_property
    def flat_mask(self):
        zs_list = [int(e) for e in np.array(self.dataset.variables['ZS'])]
        return np.array(zs_list) == self.altitude

    @property
    def study_massif_names(self) -> List[str]:
        massif_ids = np.array(self.dataset.variables['MASSIF_NUMBER'])[self.flat_mask]
        return [self.massif_number_to_massif_name[massif_id] for massif_id in massif_ids]

