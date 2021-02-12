import os
import os.path as op
import subprocess
from collections import OrderedDict
from datetime import datetime, timedelta
from enum import Enum
from typing import List

import numpy as np
from netCDF4._netCDF4 import Dataset

from extreme_data.meteo_france_data.adamont_data.adamont.adamont_variables import AbstractAdamontVariable
from extreme_data.meteo_france_data.adamont_data.adamont_gcm_rcm_couples import get_gcm_rcm_couple_adamont_to_full_name
from extreme_data.meteo_france_data.adamont_data.adamont_scenario import scenario_to_str, AdamontScenario, \
    get_year_min_and_year_max_from_scenario, get_suffix_for_the_nc_file, \
    scenario_to_real_scenarios, get_year_max, adamont_scenarios_real
from extreme_data.meteo_france_data.adamont_data.utils.utils import massif_number_to_massif_name

from extreme_data.utils import DATA_PATH

ADAMONT_PATH = op.join(DATA_PATH, 'ADAMONT')
ADAMONT_v2_PATH = op.join(DATA_PATH, 'ADAMONT_v2')
ADAMONT_v2_WEBPATH = """https://climatedata.umr-cnrm.fr/public/dcsc/projects/ADAMONT_MONTAGNE_2020/INDICATEURS_ANNUELS/alp_flat/"""

from cached_property import cached_property

from extreme_data.meteo_france_data.scm_models_data.abstract_study import AbstractStudy
from extreme_data.meteo_france_data.scm_models_data.utils import Season, FrenchRegion, french_region_to_str


class WrongYearMinOrYearMax(Exception):
    pass


class AbstractAdamontStudy(AbstractStudy):
    YEAR_MIN = 1950
    YEAR_MAX = 2100

    def __init__(self, variable_class: type, altitude: int = 1800,
                 year_min=None, year_max=None,
                 multiprocessing=True, season=Season.annual,
                 french_region=FrenchRegion.alps,
                 scenario=AdamontScenario.histo,
                 gcm_rcm_couple=('CNRM-CM5', 'ALADIN53'),
                 adamont_version=2):
        # Load the default year_min & year_max for the scenario if not specified
        year_min_scenario, year_max_scenario = get_year_min_and_year_max_from_scenario(scenario, gcm_rcm_couple)
        # Raise exception
        if year_min is None:
            year_min = year_min_scenario
        else:
            if year_min < year_min_scenario:
                raise WrongYearMinOrYearMax(
                    'year min is {} and should be larger than {} for {}'.format(year_min, year_min_scenario, gcm_rcm_couple))

        if year_max is None:
            year_max = year_max_scenario
        super().__init__(variable_class=variable_class, altitude=altitude, year_min=year_min, year_max=year_max,
                         multiprocessing=multiprocessing, season=season, french_region=french_region)
        self.adamont_version = adamont_version
        self.gcm_rcm_couple = gcm_rcm_couple
        self.gcm_rcm_full_name = get_gcm_rcm_couple_adamont_to_full_name(version=self.adamont_version)[gcm_rcm_couple]
        self.scenario = scenario
        assert issubclass(self.variable_class, AbstractAdamontVariable)
        # Assert the massif_name are in the same order
        for i, massif_name in enumerate(self.all_massif_names()):
            assert massif_name == massif_number_to_massif_name[i + 1]
        assert self.adamont_version in [1, 2]

    @property
    def variable_name(self):
        return scenario_to_str(self.scenario) + ' ' + super().variable_name

    @cached_property
    def year_to_annual_maxima(self) -> OrderedDict:
        if self.adamont_version == 1:
            return super().year_to_annual_maxima
        else:
            return self.load_year_to_annual_maxima_version_2()

    # Loading part for adamont v2

    def load_year_to_annual_maxima_version_2(self):
        year_to_annual_maxima = OrderedDict()
        for dataset in self.datasets:
            annual_maxima = np.array(dataset.variables[self.variable_class.indicator_name_for_maxima])
            assert annual_maxima.shape[1] == len(self.column_mask)
            annual_maxima = annual_maxima[:, self.column_mask]
            year_to_annual_maxima = OrderedDict()
            year_min, year_max = get_year_min_and_year_max_from_scenario(self.scenario, self.gcm_rcm_couple)
            years = list(range(year_min, year_max + 1))
            if self.scenario in adamont_scenarios_real:
                time = np.array(dataset.variables['time'])
                msg = 'len_years={} while len_time={},' \
                      'check year_min and year_max, ' \
                      'check in debug mode the time field of the daatset to see the starting date'.format(years, time)
                # # Some print to check which year are in the data
                # start = datetime(year=2005, month=8, day=1, hour=6, minute=0, second=0)
                # dates = [start + timedelta(hours=int(h)) for h in time]
                # print(["{}-{}".format(date.year-1, date.year) for date in dates])
                assert len(years) == len(time), msg
            for year, maxima in zip(years, annual_maxima):
                if self.year_min <= year <= self.year_max:
                    year_to_annual_maxima[year] = maxima
        return year_to_annual_maxima

    def _load_dataset(self, scenario):
        scenario_name = scenario_to_str(scenario)
        nc_filename = self.nc_filename_adamont_v2(scenario)
        nc_folder = op.join(ADAMONT_v2_PATH, self.variable_folder_name, scenario_name)
        nc_filepath = op.join(nc_folder, nc_filename)
        # Assert that the file is present, otherwise download it
        if not op.exists(nc_filepath):
            self._download_year_to_annual_maxima_version_2(scenario, nc_folder)
        # Load the file
        dataset = Dataset(filename=nc_filepath)
        return dataset

    def _download_year_to_annual_maxima_version_2(self, scenario, path_folder):
        scenario_name = self._scenario_to_str_adamont_v2(scenario)
        directory = self.gcm_rcm_full_name + '_' + scenario_name
        filename = self.nc_filename_adamont_v2(scenario)
        full_path = op.join(ADAMONT_v2_WEBPATH, directory, filename)
        # Download file
        request = 'wget {} -P {}'.format(full_path, path_folder)
        print(request)
        subprocess.run(request, shell=True)

    def nc_filename_adamont_v2(self, scenario):
        scenario_name = self._scenario_to_str_adamont_v2(scenario)
        return '_'.join([self.variable_class.indicator_name_for_maxima, self.gcm_rcm_full_name, scenario_name]) + '.nc'

    def _scenario_to_str_adamont_v2(self, scenario):
        scenario_name = scenario_to_str(scenario)
        if scenario is AdamontScenario.histo:
            scenario_name += 'RICAL'
        return scenario_name

    # Loading part

    @cached_property
    def ordered_years(self):
        return list(range(self.year_min, self.year_max + 1))

    def winter_years_for_each_time_step(self, real_scenario, dataset):
        year_min, year_max = get_year_min_and_year_max_from_scenario(real_scenario, self.gcm_rcm_couple)
        # It was written in the dataset  for the TIME variable that it represents
        # "'hours since 1950-08-01 06:00:00'" for the HISTO scenario
        # "'hours since 2005-08-01 06:00:00'" for the RCP scenario
        start = datetime(year=year_min - 1, month=8, day=1, hour=6, minute=0, second=0)
        hours_after_start = np.array(dataset.variables['TIME'])
        dates = [start + timedelta(hours=h) for h in hours_after_start]
        winter_years = [date.year if date.month < 8 else date.year + 1 for date in dates]
        return winter_years

    @cached_property
    def year_to_variable_object(self) -> OrderedDict:
        year_to_data_list = {}
        for year in self.ordered_years:
            year_to_data_list[year] = []
        # Load data & year list
        data_list, data_year_list = [], []
        for dataset, real_scenario in zip(self.datasets, self.adamont_real_scenarios):
            data = dataset.variables[self.variable_class.keyword()]
            data = np.array(data)
            data_list.extend(data)
            data_year_list.extend(self.winter_years_for_each_time_step(real_scenario, dataset))
            # Remark. The last winter year for the HISTO scenario correspond to 2006.
            # Thus, the last value is not taken into account
            if data_year_list[-1] > get_year_max(real_scenario, self.gcm_rcm_couple):
                data_year_list = data_year_list[:-1]
                data_list = data_list[:-1]
            assert len(data_list) == len(data_year_list)
        for year_data, data in zip(data_year_list, data_list):
            if self.year_min <= year_data <= self.year_max:
                year_to_data_list[year_data].append(data)
        # Load efficiently the variable object
        # Multiprocessing is set to False, because this is not a time consuming part
        data_list_list = [year_to_data_list[year] for year in self.ordered_years]
        year_to_variable_object = self.efficient_variable_loading(self.ordered_years, data_list_list,
                                                                  multiprocessing=False)
        return year_to_variable_object

    def load_variable_object(self, data_list):
        variable_array = np.array(data_list)
        variable_object = self.variable_class(variable_array)
        return variable_object

    @cached_property
    def flat_mask(self):
        zs_list = [int(e) for e in np.array(self.datasets[0].variables['ZS'])]
        zs_list[-10:] = [np.nan for _ in range(10)]
        if len(self.datasets) > 1:
            zs_list_bis = [int(e) for e in np.array(self.datasets[1].variables['ZS'])]
            zs_list_bis[-10:] = [np.nan for _ in range(10)]
            assert all([(a == b) or (np.isnan(a) and np.isnan(b)) for a, b in zip(zs_list, zs_list_bis)])
        return np.array(zs_list) == self.altitude

    @cached_property
    def study_massif_names(self) -> List[str]:
        massif_key = 'MASSIF_NUMBER' if self.adamont_version == 1 else 'massif_number'
        massif_ids = np.array(self.datasets[0].variables[massif_key])[self.flat_mask]
        if len(self.datasets) > 1:
            massif_ids_bis = np.array(self.datasets[1].variables[massif_key])[self.flat_mask]
            assert all(massif_ids == massif_ids_bis)
        return [massif_number_to_massif_name[massif_id] for massif_id in massif_ids]

    @cached_property
    def datasets(self):
        if self.adamont_version == 1:
            return [Dataset(file_path) for file_path in self.nc_file_paths]
        else:
            return [self._load_dataset(scenario) for scenario in self.adamont_real_scenarios]

    # PATHS

    @property
    def variable_folder_name(self):
        return self.variable_class.variable_name_for_folder_and_nc_file()

    @property
    def region_name(self):
        return french_region_to_str(self.french_region)

    @property
    def adamont_real_scenarios(self):
        return scenario_to_real_scenarios(self.scenario)

    @property
    def scenario_names(self):
        return [scenario_to_str(scenario) for scenario in self.adamont_real_scenarios]

    @property
    def nc_files_paths(self):
        return [op.join(ADAMONT_PATH, self.variable_folder_name, name) for name in self.scenario_names]

    @property
    def nc_file_paths(self):
        file_paths = []
        for scenario, scenario_name, files_path in zip(self.adamont_real_scenarios, self.scenario_names,
                                                       self.nc_files_paths):
            suffix_nc_file = get_suffix_for_the_nc_file(scenario, self.gcm_rcm_couple)
            nc_file = '{}_FORCING_{}_{}_{}_{}.nc'.format(self.variable_folder_name, self.gcm_rcm_full_name,
                                                         scenario_name,
                                                         self.region_name, suffix_nc_file)
            file_paths.append(op.join(files_path, nc_file))
        assert len(file_paths) <= 2, 'change my code to handle datasets of length larger than'
        return file_paths
