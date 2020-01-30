from datetime import datetime

import numpy as np
import os.path as op
from cached_property import cached_property
from netCDF4._netCDF4 import Dataset
from datetime import timedelta

from experiment.meteo_france_data.scm_models_data.abstract_study import AbstractStudy


class SingleSimulation(object):

    def __init__(self, nc_path, parameter, fist_year, last_year):
        self.fist_year = fist_year
        self.last_year = last_year
        self.parameter = parameter
        self.nc_path = nc_path

    @cached_property
    def dataset(self):
        return Dataset(self.nc_path)

    @cached_property
    def winter_year(self):
        start = datetime(year=1900, month=1, day=1, hour=0, minute=0, second=0)
        seconds_after_start = np.array(self.dataset.variables['TIME'])
        dates = [start + timedelta(seconds=s) for s in seconds_after_start]
        winter_year = [date.year - 1 if date.month < 8 else date.year for date in dates]
        return np.array(winter_year)

    @cached_property
    def years(self):
        return sorted([year for year in set(self.winter_year) if self.fist_year <= year <= self.last_year])

    def massif_name_and_altitude_to_return_level(self):
        return {}

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
        return dict([e.split() for e in l])

    @cached_property
    def massif_name_and_altitude_to_annual_maxima_time_series(self):
        all_values = np.array(self.dataset.variables[self.parameter])
        zs_list = [int(e) for e in np.array(self.dataset.variables['ZS'])]
        massif_number_list = np.array(self.dataset.variables['MASSIF_NUMBER'])
        massif_name_list = [self.massif_number_to_massif_name[str(n)] for n in massif_number_list]
        d = {}
        for year in self.years:
            indexes = np.where(self.winter_year == year)[0]
            winter_values = all_values[indexes, 0, :]
            assert len(winter_values) in [365, 366]
            for time_serie, zs, massif_name in zip(winter_values.transpose(), zs_list, massif_name_list):
                # print(zs, massif_name, len(time_serie))
                d[(massif_name, zs)] = time_serie
        return d

    @cached_property
    def massif_name_and_altitude_to_average_maxima(self):
        return {t: np.mean(s) for t, s in self.massif_name_and_altitude_to_annual_maxima_time_series.items()}
