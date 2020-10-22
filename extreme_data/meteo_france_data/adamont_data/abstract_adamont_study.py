import os
import os.path as op
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import List

import numpy as np
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import AbstractAdamontVariable
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import scenario_to_str, AdamontScenario, \
    get_year_min_and_year_max_from_scenario, gcm_rcm_couple_to_full_name, get_suffix_for_the_nc_file
from extreme_data.meteo_france_data.adamont_data.utils.utils import massif_number_to_massif_name

ADAMONT_PATH = r"/home/erwan/Documents/projects/spatiotemporalextremes/local/spatio_temporal_datasets/ADAMONT"

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion, french_region_to_str


class AbstractAdamontStudy(AbstractStudy):
    YEAR_MIN = 1950
    YEAR_MAX = 2100

    def __init__(self, variable_class: type, altitude: int = 1800,
                 year_min=None, year_max=None,
                 multiprocessing=True, season=Season.annual,
                 french_region=FrenchRegion.alps,
                 scenario=AdamontScenario.histo, gcm_rcm_couple=('CNRM-CM5', 'ALADIN53')):
        # Load the default year_min & year_max for the scenario if not specified
        if year_min is None and year_max is None:
            year_min, year_max = get_year_min_and_year_max_from_scenario(scenario, gcm_rcm_couple)
        super().__init__(variable_class=variable_class, altitude=altitude, year_min=year_min, year_max=year_max,
                         multiprocessing=multiprocessing, season=season, french_region=french_region)
        self.gcm_rcm_couple = gcm_rcm_couple
        self.gcm_rcm_full_name = gcm_rcm_couple_to_full_name[gcm_rcm_couple]
        self.scenario = scenario
        assert issubclass(self.variable_class, AbstractAdamontVariable)
        # Assert the massif_name are in the same order
        for i, massif_name in enumerate(self.all_massif_names()):
            assert massif_name == massif_number_to_massif_name[i + 1]

    @property
    def variable_name(self):
        return scenario_to_str(self.scenario) + ' ' + super().variable_name

    # Loading part



    @cached_property
    def ordered_years(self):
        return list(range(self.year_min, self.year_max + 1))

    @cached_property
    def winter_year_for_each_time_step(self):
        start = datetime(year=self.year_min - 1, month=8, day=1, hour=6, minute=0, second=0)
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
        data_list = self.dataset.variables[self.variable_class.keyword()]
        data_year_list = self.winter_year_for_each_time_step
        assert len(data_list) == len(data_year_list)
        for year_data, data in zip(data_year_list, data_list):
            if self.year_min <= year_data <= self.year_max:
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

    @cached_property
    def study_massif_names(self) -> List[str]:
        massif_ids = np.array(self.dataset.variables['MASSIF_NUMBER'])[self.flat_mask]
        return [massif_number_to_massif_name[massif_id] for massif_id in massif_ids]

    @cached_property
    def dataset(self):
        return Dataset(self.nc_file_path)

    # PATHS

    @property
    def variable_folder_name(self):
        return self.variable_class.variable_name_for_folder_and_nc_file()

    @property
    def scenario_name(self):
        return scenario_to_str(self.scenario)

    @property
    def region_name(self):
        return french_region_to_str(self.french_region)

    @property
    def nc_files_path(self):
        return op.join(ADAMONT_PATH, self.variable_folder_name, self.scenario_name)

    @property
    def nc_file_path(self):
        suffix_nc_file = get_suffix_for_the_nc_file(self.scenario, self.gcm_rcm_couple)
        nc_file = '{}_FORCING_{}_{}_{}_{}.nc'.format(self.variable_folder_name, self.gcm_rcm_full_name, self.scenario_name,
                                                     self.region_name, suffix_nc_file)
        return op.join(self.nc_files_path, nc_file)
